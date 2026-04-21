"""Welcome/menu flow plugin.

This handler acts as the router after login. It responds to main-menu intents
and hands off to encouragement, quiz, or chat flows.
"""

import runtimeFlowPlugins
from .encouragementGenerator import encouragement_switch

# Centralized main menu message for consistent UX across all return-to-menu flows.
MAIN_MENU_WELCOME_MESSAGE = (
    "Welcome to the main menu! You can ask for encouragement, take a quiz, or chat with me! "
    "What would you like to do?"
)

# Transition message shown when a handler is returning to main menu.
RETURN_TO_MENU_MESSAGE = "Returning to main menu."


def get_main_menu_message() -> str:
    """Return the standard main menu welcome message for consistent UX."""
    return MAIN_MENU_WELCOME_MESSAGE


def get_return_to_menu_message() -> str:
    """Return the transition message shown before returning to main menu."""
    return RETURN_TO_MENU_MESSAGE

@runtimeFlowPlugins.register("WelcomeHandler")
def welcome_handler(state, meta, inputText, predictedIntent):
    """Route menu-level intents and return standardized flow outcomes."""
    #defaults: 
    nextHandler = "WelcomeHandler"
    nextResponse = ""
    nextState = state
    nextMeta = meta

    # intent points to different flows:
    # encouragement -> call encouragement for a encouragement response and then come back
    # quiz -> call and hand off to quiz flow, and then come back to main menu after quiz is done
    # chat -> call and hand off to chat flow, and then come back to main menu after user says they want to exit the chat
    # capture any handoffs and do not handoff from here unless it is from a success state after this intent has handled once 
    if state == "passoff":
        nextResponse = get_main_menu_message()
        nextState = "success"
        return {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}

    if predictedIntent == "encouragement":
        nextResponse = encouragement_switch("any")
    elif predictedIntent == "quiz":
        nextHandler = "QuizHandler"
        nextState = "passoff"
    elif predictedIntent == "chat":
        nextHandler = "ChatHandler"
        nextState = "passoff"
    else:
        nextResponse = get_main_menu_message()
        nextState = "success"
    return {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}
