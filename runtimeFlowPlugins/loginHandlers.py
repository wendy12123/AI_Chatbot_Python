"""Login flow plugin.

This handler guides the user through username/password states and verifies
credentials using salted SHA-256 hashes from YAML files in `runtimeInfo/userInfo`.
"""

import runtimeFlowPlugins
from pathlib import Path
from enum import Enum, auto
import hashlib
import yaml

BASEPATH = Path(__file__).resolve().parent.parent
USERFILEPATH = BASEPATH / "runtimeInfo" / "userInfo"

class LoginResult(Enum):
    SUCCESS = auto()
    NO_USER = auto()
    WRONG_PASSWORD = auto()

@runtimeFlowPlugins.register("LoginHandler")
def login_handler(state, meta, inputText, predictedIntent):
    """Process one login step and return the next state/handler outcome.

    Parameters:
    - state: current login state string.
    - meta: shared conversation metadata dict.
    - inputText: latest user input text.
    - predictedIntent: current predicted intent (unused in login states).

    Returns:
    - dict with keys `response`, `next_handler`, `next_state`, `meta_update`.
    """
    #defaults: 
    nextHandler = "LoginHandler"
    nextResponse = ""
    nextState = state
    nextMeta = meta
    #predictedIntent should be none for the first call, and then it will be updated based on the user's input and the flow's logic
    if state == "start" or state == "passoff":
        nextResponse = "Hello! I'm Snackie from CPCE SPEED and I'm here to assist you in learning SEHS4678! You can chat with me, have quizzes or get encouragements! \n Please enter your username to get started, or type 'register' to create a new account."
        nextState = "awaiting_username"
        return {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}
    
    if state == "awaiting_username":
        if inputText.strip().lower() == 'register':
            return {
                "response": "", # RegistrationHandler response will be provided
                "next_handler": "RegistrationHandler", 
                "next_state": "passoff",
                "meta_update": meta
            }
        nextResponse = "Please enter your password."
        nextState = "awaiting_password"
        nextMeta["username"] = inputText
        return {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}
    if state == "awaiting_password":
        username = meta.get("username")
        password = inputText
        login_result = passwordChecker(username, password)
        if login_result == LoginResult.SUCCESS:
            nextState = "passoff"
            nextHandler = "WelcomeHandler"  # Transition to the main menu handler after successful login
        elif login_result == LoginResult.NO_USER:
            nextResponse = "No such user found. Please enter your username again."
            nextState = "awaiting_username"
        elif login_result == LoginResult.WRONG_PASSWORD:
            nextResponse = "Incorrect password. Please enter your password again."
            nextState = "awaiting_password"

    return {"response": nextResponse, "next_handler": nextHandler, "next_state": nextState, "meta_update": nextMeta}

def passwordChecker(username, password):
    """Verify username/password against salted hash in the user YAML file."""
    userFile = USERFILEPATH / f"{username}.yaml"
    if not userFile.exists():
        return LoginResult.NO_USER
    with open(userFile, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        # passwords are sha256 hashed with salt prepended.
        salted_input = password + data["salt"]
        hashed_input = hashlib.sha256(salted_input.encode()).hexdigest()
        if hashed_input == data["hashed_password"]:
            return LoginResult.SUCCESS
        else:
            return LoginResult.WRONG_PASSWORD
        
@runtimeFlowPlugins.register("SettingHandler")
def setting_handler(state, meta, inputText, predictedIntent):
    """Placeholder for a settings handler that could allow users to change their password or other preferences."""
    username = meta.get("username")
    nextHandler = "SettingHandler"
    nextMeta = meta
    if state == "verify_old_password":
        if inputText.strip().lower() == "exit":
            return {"response": "Password change cancelled. Returning to main menu.", "next_handler": "WelcomeHandler", "next_state": "passoff", "meta_update": nextMeta}
        if passwordChecker(username, inputText) == LoginResult.SUCCESS:
            return {"response": "Verification successful. Please enter your new password:", "next_handler": nextHandler, "next_state": "enter_new_password", "meta_update": nextMeta}
        else:
            return {"response": "Incorrect password. Please try again or type 'exit' to cancel: ", "next_handler": nextHandler, "next_state": "verify_old_password", "meta_update": nextMeta}
    if state == "enter_new_password":
        new_password = inputText
        update_password(username, new_password)
        return {"response": "Your password has been updated successfully.", "next_handler": "WelcomeHandler", "next_state": "passoff", "meta_update": nextMeta}
    return {"response": "Something went wrong. Returning to menu.", "next_handler": "WelcomeHandler", "next_state": "passoff", "meta_update": nextMeta}
    
def update_password(username, new_password):
    userFile = USERFILEPATH / f"{username}.yaml"
    with open(userFile, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    salt = data["salt"]
    salted_input = new_password + salt
    new_hashed_password = hashlib.sha256(salted_input.encode()).hexdigest()
    data["hashed_password"] = new_hashed_password
    with open(userFile, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)