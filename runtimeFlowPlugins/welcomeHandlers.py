import runtimeFlowPlugins
from .encouragementGenerator import encouragement_switch

@runtimeFlowPlugins.register("WelcomeHandler")
def welcome_handler(state, meta, inputText, predictedIntent):
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
        nextResponse = "Welcome to the main menu! You can ask for encouragement, take a quiz, or chat with me! What would you like to do?"
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
        nextResponse = "Welcome to the main menu! You can ask for encouragement, take a quiz, or chat with me! What would you like to do?"
        nextState = "success"
    return {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}
