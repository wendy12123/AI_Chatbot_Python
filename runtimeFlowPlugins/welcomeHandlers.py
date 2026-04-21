"""Welcome/menu flow plugin.

This handler acts as the router after login. It responds to main-menu intents
and hands off to encouragement, quiz, or chat flows.
"""

import runtimeFlowPlugins
from .encouragementGenerator import encouragement_switch

# Centralized main menu message for consistent UX across all return-to-menu flows.
MAIN_MENU_WELCOME_MESSAGE = (
    "Welcome to the main menu! You can ask for encouragement, take a quiz, or chat with me! "
    "If you want to change your password, just type 'change password'."
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
        nextResponse = "Welcome to the main menu! You can ask for encouragement, take a quiz, or chat with me! If you want to change your password, just type 'change password'. What would you like to do? Or if you want to end our conversation, just type 'exit'."
        nextState = "success"
        return {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}
    
    if state == "return_to_menu": #global catch-all state to return to main menu
        return {
            "response": "",
            "next_handler": "WelcomeHandler",
            "next_state": "passoff", 
            "meta_update": meta
        }
        
    if state == "confirming_exit":
        user_reply = inputText.strip().lower()
        if user_reply == "yes":
            return {"response": "See you later!", "next_handler": "LoginHandler", "next_state": "start", "meta_update": nextMeta}
        elif user_reply == "no":
            nextResponse = "Welcome to the main menu! You can ask for encouragement, take a quiz, or chat with me! If you want to change your password, just type 'change password'. What would you like to do? Or if you want to end our conversation, just type 'exit'."
            return {"response": nextResponse, "next_handler": nextHandler, "next_state": "success", "meta_update": nextMeta}
        else:
            return {"response": "Please type 'yes' to confirm or 'no' to cancel.", "next_handler": nextHandler, "next_state": "confirming_exit", "meta_update": nextMeta}

    if state == "success":
        if inputText.strip().lower() == "change password":
            return {
                "response": "To change your password, please enter your current password:",
                "next_handler": "SettingHandler",
                "next_state": "verify_old_password",
                "meta_update": meta
            }

        elif predictedIntent == "exit":
            # the "exit" action is determined based on the role
            # retrieve the role from the meta list; if not found, default to student.
            user_role = meta.get("role", "student") 
            
            # based on the role, determine the appropriate exit behavior
            if user_role == "supervisor":
                # if the user is a supervisor, "exit" means "return to admin panel"
                return {
                    "response": "Returning to Supervisor Admin Panel...",
                    "next_handler": "SupervisorHandler", # go back to supervisor handler
                    "next_state": "passoff", 
                    "meta_update": meta
                }
            else:
                # if the user is a student, "exit" means "end conversation"
                return {
                    "response": "Are you sure you want to end our conversation? (Type 'yes' to confirm, 'no' to cancel)",
                    "next_handler": "WelcomeHandler",
                    "next_state": "confirming_exit",
                    "meta_update": meta
                }

        elif predictedIntent == "encouragement":
            return {
                "response": encouragement_switch("any") + "\n\n" + get_main_menu_message(),
                "next_handler": "WelcomeHandler",
                "next_state": "success",
                "meta_update": meta
            }

        elif predictedIntent == "quiz":
            return {"response": "", "next_handler": "QuizHandler", "next_state": "passoff", "meta_update": meta}
            
        elif predictedIntent == "chat":
            return {"response": "", "next_handler": "ChatHandler", "next_state": "passoff", "meta_update": meta}
            
        else:
            return {
                "response": "Sorry, I didn't understand. " + get_main_menu_message(),
                "next_handler": "WelcomeHandler",
                "next_state": "success",
                "meta_update": meta
            }
