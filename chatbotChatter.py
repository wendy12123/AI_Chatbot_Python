"""Runtime entry script for the Support Chatbot.

This script runs the terminal chat loop, predicts intents from user text,
dispatches flow handlers from `runtimeFlowPlugins`, and applies handler
handoff state transitions (`next_handler`, `next_state`, `meta_update`).

How to run:
    python chatbotChatter.py
"""

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import datetime
import time
import sys
from typing import Any
import os
from dotenv import load_dotenv
load_dotenv()

DEBUGGING = True

# Submodules in runtimeSubmodules/
from runtimeSubmodules.chatbotNLP import clean_up_sentence, bow, predict_class
from runtimeSubmodules.chatbotVisual import typing_effect, print_timestamp
from traintimeSubmodules.intentLoader import load_intents
import runtimeFlowPlugins

# ---------------------------
# Chatbot NLP Functions
# ---------------------------

def get_response(intents_list):
    """Pick a canned response for the top predicted intent.

    Parameters:
    - intents_list: list of intent predictions from the classifier.

    Returns:
    - str: one response line for the best-matching intent.
    """
    if not intents_list:
        return "I'm not sure I understand. Could you rephrase?"

    tag = intents_list[0]['intent']
    list_of_intents = corpus['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return "Sorry, something went wrong."


def chatbot_response(msg):
    """Generate a simple response from classifier output only."""
    ints = predict_class(msg)
    return get_response(ints)


def chatbot_callFlows(handler, state, meta, inputText, predictedIntent) -> dict[str, Any]:
    """Call the current flow plugin and return the standardized outcome object."""
    # call flow plugins based on handler input to get response text and next states
    try:
        plugin = runtimeFlowPlugins.require(handler)
        outcomes = plugin(state, meta, inputText, predictedIntent)
        # outcomes is {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}
        return outcomes
    except ValueError as e:
        return {
            "response": f"Error: {e}",
            "next_handler": handler,
            "next_state": state,
            "meta_update": meta,
        }


# ---------------------------
# UI Functions
# ---------------------------

def print_interface():
    """Print a minimal terminal banner before chat starts."""
    # TODO: Add ASCII art or a welcome message here if desired, do it later after working with the plugins
    print("=" * 50)
    print("AI Chatbot")
    print("=" * 50)
    print("Type 'quit', 'exit', or 'bye' to stop.")
    print()

# ---------------------------
# The main program
# ---------------------------

corpus = load_intents()

# Start chatting
print_interface()

handler = "LoginHandler"
state = "start"
meta = {}
text = ""
predictedIntent = ""

while True:
    try:
        if handler == "LoginHandler" and state == "start":
            # For the first call, we want to trigger the LoginHandler's start state to send the welcome message and prompt for username, so we can pass an empty string as inputText and predictedIntent since they are not used in that state.
            message = ""
            predictedIntent = ""
        else:
            message = input(f"[{print_timestamp()}] You: ")

            if message.strip() == "":
                continue

            # get intents
            intentList = predict_class(message)
            predictedIntent = intentList[0]['intent'] if intentList else "Not Detected"

        if DEBUGGING:
            print(f"[{print_timestamp()}] Predicted intent: {predictedIntent}")

        #response = chatbot_response(message)
        # outcomes is {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}
        outcomes = chatbot_callFlows(handler, state, meta, message, predictedIntent)
        response_chain = []
        if outcomes.get("response"):
            response_chain.append(outcomes["response"])

        # Auto-dispatch internal handoff states without waiting for new user input.
        passoff_guard = 0
        while outcomes["next_state"] == "passoff" and passoff_guard < 10:
            handler = outcomes["next_handler"] if outcomes["next_handler"] is not None else handler
            state = outcomes["next_state"] if outcomes["next_state"] is not None else state
            meta = outcomes["meta_update"] if outcomes["meta_update"] is not None else meta

            passoff_guard += 1
            outcomes = chatbot_callFlows(handler, state, meta, "", "")
            if outcomes.get("response"):
                response_chain.append(outcomes["response"])

        if outcomes["next_state"] == "passoff" and passoff_guard >= 10:
            print(f"[{print_timestamp()}] Bot: Internal handoff loop limit reached.")
            break

        if response_chain:
            print(f"[{print_timestamp()}] Bot: ", end="")
            typing_effect("\n\n".join(response_chain))

        # update handler, state, meta for the next round based on plugin outcomes
        handler = outcomes["next_handler"] if outcomes["next_handler"] is not None else handler
        state = outcomes["next_state"] if outcomes["next_state"] is not None else state
        meta = outcomes["meta_update"] if outcomes["meta_update"] is not None else meta

        if handler == "Main" and state == "exit":
            print("\nBot: Goodbye! Have a great day.")
            break

    except KeyboardInterrupt:
        print("\n\nBot: Chat ended. Goodbye :-)")
        break

    except Exception as e:
        print(f"\n\nBot: An error occurred: {e}")
        if DEBUGGING:
            import traceback
            traceback.print_exc()
        break
