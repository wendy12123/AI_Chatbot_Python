# runtimeFlowPlugins/supervisorHandlers.py

import runtimeFlowPlugins
from pathlib import Path
import yaml
import os

BASEPATH = Path(__file__).resolve().parent.parent
USERINFO_DIR = BASEPATH / "runtimeInfo" / "userInfo"

# --- used to scan and analyze student data ---

def get_all_student_data():
    # Scan the userInfo directory for all YAML files, read them, and filter out those with the role of "student". Return a list of student data dictionaries
    student_data = []
    for user_file in USERINFO_DIR.glob("*.yaml"):
        with open(user_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            # Ensure the file contains a role field with the value "student"
            if data and data.get("role") == "student":
                student_data.append(data)
    return student_data

def calculate_average_scores():
    """calculate the average test score for all students"""
    students = get_all_student_data()
    total_score = 0
    total_questions = 0
    students_with_scores = 0

    for student in students:
        progress = student.get("quiz_progress", {})
        if progress:
            students_with_scores += 1
            for quiz in progress.values():
                total_score += quiz.get("score", 0)
                total_questions += quiz.get("total", 0)

    if total_questions == 0:
        return "No quiz data available to calculate averages."
    
    average = (total_score / total_questions) * 100
    return f"Overall average score for all students: {average:.2f}% ({total_score}/{total_questions} correct)."

def format_all_student_scores():
    # format all students grades for easy display
    students = get_all_student_data()
    if not students:
        return "No student data found."

    response_lines = ["--- All Student Scores ---"]
    for student in students:
        username = student.get("username", "Unknown User")
        progress = student.get("quiz_progress", {})
        if not progress:
            response_lines.append(f"\nUser: {username}\n- No quiz attempts yet.")
        else:
            response_lines.append(f"\nUser: {username}")
            for quiz_key, quiz_data in progress.items():
                name = quiz_data.get("name", quiz_key)
                score = quiz_data.get("score", "N/A")
                total = quiz_data.get("total", "N/A")
                response_lines.append(f"- {name}: {score}/{total}")
    
    response_lines.append("\nType 'menu' to return to the admin panel.")
    return "\n".join(response_lines)


# --- Supervisor Handler ---
@runtimeFlowPlugins.register("SupervisorHandler")
def supervisor_handler(state, meta, inputText, predictedIntent):
    """This handler is designed for supervisors (instructors) to view student performance data."""

    if state == "passoff":
        avg_scores_summary = calculate_average_scores()
        response = (
            "Welcome to the Supervisor Admin Panel.\n\n"
            f"--- Dashboard Summary ---\n{avg_scores_summary}\n\n"
            "What would you like to do?\n"
            "1. View all student scores\n"
            "2. Enter Student Main Menu\n"
            "3. Exit"
        )
        return {
            "response": response,
            "next_handler": "SupervisorHandler",
            "next_state": "awaiting_choice",
            "meta_update": meta
        }
        
    if state == "awaiting_choice":
        choice = inputText.strip()
        if choice == "1" or "score" in choice:
            scores_report = format_all_student_scores()
            return {
                "response": scores_report,
                "next_handler": "SupervisorHandler",
                "next_state": "awaiting_choice", # after viewing scores, still stay in the admin panel
                "meta_update": meta
            }
        elif choice == "2" or "student menu" in choice:
            # give supervisors access to the student main menu, but their role in meta is still "supervisor", so they will see the supervisor-specific options in the main menu.
            return {
                "response": "Entering student main menu...",
                "next_handler": "WelcomeHandler",
                "next_state": "passoff", 
                "meta_update": meta # role='supervisor' is preserved in meta
            }
        
        elif choice == "3" or "exit" in choice:
            return {"response": "You have been logged out. Goodbye, Supervisor!", "next_handler": "LoginHandler", "next_state": "start", "meta_update": {}}
        elif "menu" in choice: # return to admin panel menu
             return {"response": "", "next_handler": "SupervisorHandler", "next_state": "passoff", "meta_update": meta}
        else:
            return {
                "response": "Invalid choice. Please select 1 or 2 or 3.",
                "next_handler": "SupervisorHandler",
                "next_state": "awaiting_choice",
                "meta_update": meta
            }
            
    return {
        "response": "Sorry, an error occurred in the admin panel. Returning to dashboard.",
        "next_handler": "SupervisorHandler",
        "next_state": "passoff",
        "meta_update": meta
    }
