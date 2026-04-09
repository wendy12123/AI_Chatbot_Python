# Support Chatbot Server

## What this project is
This project is a local, plugin-driven support chatbot for SEHS4678 learning activities. It combines:
- Intent classification (TensorFlow/Keras model over bag-of-words features)
- Runtime flow handlers (login, welcome menu, quiz, chat, encouragement)
- YAML-based content and user records

## Why it is designed this way
The architecture uses simple, explainable building blocks so it is easy to study and maintain:
- Bag-of-words + lemmatization for intent detection: easy to understand, fast to train for small datasets.
- Flow plugins registered by name: each conversational feature is isolated in one handler, so extending behavior is safer.
- YAML content files: non-code updates for intents, quizzes, encouragements, and user progress.

This design trades deep language understanding for simplicity and control, which is usually a good fit for coursework projects and small domain bots.

## Main capabilities
- User login with salted SHA-256 password check.
- Main menu routing to:
  - encouragement responses
  - quiz flow with scoring and progress save
  - chat flow with lightweight local retrieval from runtime info files
- Exit and handoff states between flows without restarting the program.

## Project structure
- chatbotChatter.py: Runtime entry script. Runs the chat loop, predicts intents, and dispatches handlers.
- chatbotIntentTrainer.py: Training script. Builds intent model and saves runtime artifacts.
- runtimeFlowPlugins/: Conversation flow plugins and registry.
  - __init__.py: Plugin registry and autoload.
  - loginHandlers.py: Username/password flow and validation.
  - welcomeHandlers.py: Main menu and intent-based routing.
  - quizHandlers.py: Quiz set selection, question handling, scoring, persistence.
  - chatHandlers.py: Local retrieval chat over runtime knowledge files.
  - encouragementGenerator.py: Encouragement message selection from YAML.
- runtimeSubmodules/: Shared runtime helpers.
  - chatbotNLP.py: Tokenization, bag-of-words, intent prediction.
  - chatbotVisual.py: Typing effect and timestamps.
- traintimeSubmodules/: Training-time helpers.
  - intentLoader.py: Intent YAML loader and validation.
- runtimeInfo/: Runtime content.
  - Intents/: Intent definitions for classifier training and runtime responses.
  - quiz/: Quiz set definitions.
  - encouragement/: Encouragement message pools.
  - userInfo/: User profile records, password hash/salt, quiz progress.
- runtimeModels/: Generated runtime artifacts (model and vocabulary/class pickles).

## Data format reference
### 1) Intent file (runtimeInfo/Intents/*.yaml)
Required keys:
- name: intent name
- patterns: example user utterances
- responses: canned replies

Example:
name: quiz
patterns:
  - "Start quiz"
responses:
  - "Sure, I can help you start a quiz."

### 2) Quiz set file (runtimeInfo/quiz/*.yaml)
Common keys:
- name
- description
- questions: list of question objects

Question fields:
- question: prompt text
- type: fill-in-the-blank or multiple-choice
- answer: accepted answer(s) or option number
- options: required for multiple-choice
- capitalization: regard or disregard (for fill-in-the-blank)
- reason: explanation shown when answer is wrong

### 3) User file (runtimeInfo/userInfo/*.yaml)
Common keys:
- username
- salt
- hashed_password
- quiz_progress: completion and score records by set

## Setup and run
## Prerequisites
- Python 3.10+ recommended
- pip package manager

Install core dependencies:
pip install tensorflow keras nltk numpy pyyaml matplotlib

Notes:
- The runtime and training scripts call nltk.download(...) automatically.
- If your environment is strict/offline, preinstall NLTK data manually.

## Train intent model
Run from this folder:
python chatbotIntentTrainer.py

This generates:
- runtimeModels/intent_model.keras
- runtimeModels/words.pkl
- runtimeModels/classes.pkl

## Start chatbot runtime
Run:
python chatbotChatter.py

Expected flow:
1. Bot asks for username.
2. Bot asks for password.
3. On success, bot hands off to WelcomeHandler main menu.
4. User can choose encouragement, quiz, or chat.

## How flow handoff works
Each handler returns a dictionary with:
- response
- next_handler
- next_state
- meta_update

The runtime loop updates current handler/state/meta using this outcome. A passoff state triggers internal auto-dispatch so flows can transition without waiting for extra user input.

## Extension guide
### Add a new flow plugin
1. Create a new file under runtimeFlowPlugins/.
2. Define a function and register it with @runtimeFlowPlugins.register("YourHandlerName").
3. Keep function signature consistent:
   (state, meta, inputText, predictedIntent)
4. Return the standard outcome dictionary.
5. Route to your new handler from welcomeHandlers.py (or another handler).

Because plugins auto-load from the package, new modules are discovered at startup.

### Add new intents
1. Add YAML under runtimeInfo/Intents/ with required keys.
2. Retrain using chatbotIntentTrainer.py.
3. Verify runtimeModels files are updated.

### Add quiz content
1. Add a new YAML set in runtimeInfo/quiz/.
2. Ensure each question has the required fields.
3. QuizHandler will include new sets automatically.

## Important considerations and edge cases
- Runtime artifact dependency: chatbotNLP.py loads model and pickles on import. Runtime will fail if files are missing.
- Login security logging: loginHandlers.py currently prints salted input during password checking. Remove this print in production.
- Input data validation: malformed intent YAML raises ValueError by design to avoid silent misbehavior.
- Quiz matching: set selection supports exact, numeric, alias-like, and fuzzy matching, but ambiguous user text can still pick the wrong set.
- Local retrieval chat: chat mode retrieves snippets from local runtime info files and does not call external APIs.

## Typical development workflow
1. Edit intents, quiz sets, or encouragement data in runtimeInfo/.
2. If intents changed, retrain model.
3. Run chatbot runtime and test full flow:
   login -> welcome -> feature flow -> return to menu.
4. Verify updated user progress in runtimeInfo/userInfo/*.yaml after quizzes.

## Troubleshooting quick list
- Model file not found:
  Train first with chatbotIntentTrainer.py.
- NLTK resource error:
  Run once with internet, or manually install punkt/wordnet resources.
- Login always fails:
  Confirm username file exists and hash is generated with password + salt (same order as runtime checker).
- New plugin not called:
  Confirm file is under runtimeFlowPlugins and function is registered with the exact handler name used in handoff.

## Suggested cleanup for production
- Add a pinned requirements.txt in this folder.
- Remove sensitive debug prints from login.
- Add unit tests for plugin transitions and quiz answer evaluation.
- Add schema checks for quiz and user YAML files.


add:
ragknowledge: is made by google gemini on lecture notes