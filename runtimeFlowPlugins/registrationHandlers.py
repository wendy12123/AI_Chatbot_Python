# runtimeFlowPlugins/registrationHandlers.py

import runtimeFlowPlugins
from pathlib import Path
import hashlib
import os
import yaml

BASEPATH = Path(__file__).resolve().parent.parent
USERFILEPATH = BASEPATH / "runtimeInfo" / "userInfo"

def create_new_user(username, password):
    #check if the user exists; if not, create a new user YAML profile.
    user_file = USERFILEPATH / f"{username}.yaml"
    if user_file.exists():
        return False # indicates that the user already exists

    # random salt
    salt = os.urandom(16).hex()
    
    # Combining password and salt
    salted_input = password + salt
    hashed_password = hashlib.sha256(salted_input.encode()).hexdigest()
    
    # Prepare user data to be written
    new_user_data = {
        "username": username,
        "salt": salt,
        "hashed_password": hashed_password,
        "quiz_progress": {} # The quiz progress for new users is empty
    }
    
    # Write a new YAML file
    with open(user_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_user_data, f, sort_keys=False)
        
    return True # if true, successful creation

@runtimeFlowPlugins.register("RegistrationHandler")
def registration_handler(state, meta, inputText, predictedIntent):
    
    # the entry point for the registration process
    if state == "passoff":
        return {
            "response": "Let's create a new account. Please choose a username.",
            "next_handler": "RegistrationHandler",
            "next_state": "awaiting_username",
            "meta_update": meta
        }

    # waiting user to input name
    if state == "awaiting_username":
        username = inputText.strip()
        if not username: # make sure the username is not empty.
            return {
                "response": "Username cannot be empty. Please choose a valid username.",
                "next_handler": "RegistrationHandler",
                "next_state": "awaiting_username",
                "meta_update": meta
            }
        
        # Check if the username already exists
        user_file = USERFILEPATH / f"{username}.yaml"
        if user_file.exists():
            return {
                "response": "This username is already taken. Please choose another one.",
                "next_handler": "RegistrationHandler",
                "next_state": "awaiting_username",
                "meta_update": meta
            }
        
        # if username is available, save it to meta and ask for a password
        meta['new_username'] = username
        return {
            "response": f"Great, '{username}' is available. Now, please set your password.",
            "next_handler": "RegistrationHandler",
            "next_state": "awaiting_password",
            "meta_update": meta
        }
        
    if state == "awaiting_password":
        password = inputText
        if len(password) < 6:
            return {
                "response": "Password should be at least 6 characters long. Please try again.",
                "next_handler": "RegistrationHandler",
                "next_state": "awaiting_password",
                "meta_update": meta
            }
        
        # first time password is temporarily stored in the meta
        meta['new_password'] = password
        
        # enter the password again for confirmation
        return {
            "response": "Thank you. Please type your password again to confirm.",
            "next_handler": "RegistrationHandler",
            "next_state": "confirming_password", 
            "meta_update": meta
        }

    # compare the two passwords and create the new user if they match
    if state == "confirming_password":
        confirmed_password = inputText
        first_password = meta.get('new_password')
        
        # compare the two passwords 
        if confirmed_password == first_password:
            # passwords match, create user account
            username = meta.get('new_username')
            success = create_new_user(username, confirmed_password) 
            
            if success:
                # passwords match, execute user creation logic
                return {
                    "response": "Passwords matched. Account created successfully! You can now log in with your new credentials.",
                    "next_handler": "LoginHandler",
                    "next_state": "passoff",
                    "meta_update": {} # clear meta tags to begin a clean login process
                }
            else:
                return {
                    "response": "Something went wrong while creating your account. Let's try again.",
                    "next_handler": "RegistrationHandler",
                    "next_state": "awaiting_username",
                    "meta_update": meta
                }
        else:
            # password mismatch
            return {
                "response": "The passwords do not match. Let's try again. Please set a new password.",
                "next_handler": "RegistrationHandler",
                "next_state": "awaiting_password",
                "meta_update": meta
            }

    return {
        "response": "Sorry, I got lost in the registration process. Let's start over.",
        "next_handler": "RegistrationHandler",
        "next_state": "passoff",
        "meta_update": meta
    }
