# Support Chatbot Server

This folder contains a local, plugin-driven support chatbot for SEHS4678 learning activities. The current setup uses a small TensorFlow/Keras intent model, YAML-driven flows, and local retrieval content for chat support.

## Current setup

The project is built around a few simple parts:

- Intent classification with bag-of-words features and lemmatization.
- Runtime flow plugins for login, welcome/menu routing, quiz handling, chat, and encouragement.
- YAML content for intents, quizzes, encouragements, and user data.
- Local retrieval knowledge under `runtimeInfo/ragKnowledge/`.

This design keeps the system easy to inspect and update for coursework use. It favors control and clarity over deep language understanding.

## Technical specification

### Use of LLM

An LLM is used only during content preparation for the retrieval knowledge base. The `runtimeInfo/ragKnowledge/` material was prepared from lecture notes with Google Gemini, then stored as local markdown or text files for the chatbot to retrieve at runtime.

### Runtime behavior

The chatbot itself does not call an external LLM when responding to users. Instead, it uses:

- a trained intent classifier for menu and flow routing,
- rule-based plugin handlers for conversation logic,
- local retrieval over `runtimeInfo/ragKnowledge/` for topic explanations.

### Technical implications

- No API key is required for normal chatbot execution.
- Responses are deterministic enough for coursework review and debugging.
- Knowledge updates are handled by editing local files rather than retraining or calling a hosted model.
- The quality of retrieval depends on the clarity and structure of the source notes.

## What it does

- Verifies users with salted SHA-256 password checks.
- Routes users through the main menu after login.
- Supports encouragement replies, quiz sessions, and local chat retrieval.
- Saves quiz progress in the user YAML files.
- Uses handoff states so flows can move between handlers without restarting the program.

## Folder overview

- `chatbotChatter.py`: Runtime entry point that runs the conversation loop.
- `chatbotIntentTrainer.py`: Training script that builds the intent model and runtime artifacts.
- `runtimeFlowPlugins/`: Runtime handler plugins and registry.
  - `loginHandlers.py`: Login flow and credential validation.
  - `welcomeHandlers.py`: Main menu and routing.
  - `quizHandlers.py`: Quiz selection, scoring, and progress saving.
  - `chatHandlers.py`: Local retrieval chat over runtime knowledge files.
  - `encouragementGenerator.py`: Encouragement message selection.
- `runtimeSubmodules/`: Shared runtime helpers.
  - `chatbotNLP.py`: Tokenization, bag-of-words, and intent prediction.
  - `chatbotVisual.py`: Typing and display helpers.
- `traintimeSubmodules/`: Training-time helpers.
  - `intentLoader.py`: Intent YAML loading and validation.
- `runtimeInfo/`: Runtime content.
  - `Intents/`: Intent definitions for training and runtime responses.
  - `quiz/`: Quiz set definitions.
  - `encouragement/`: Encouragement message pools.
  - `userInfo/`: User profile records, password hash/salt, and quiz progress.
  - `ragKnowledge/`: Retrieval knowledge files used by chat. This content was prepared from lecture notes with Google Gemini.
- `chatbotHeadlessTelegramBot.py`: Telegram runtime that uses the same plugin flow logic as the terminal chatbot.
- `headlessLog/`: JSONL logs for Telegram incoming/outgoing messages.
- `runtimeModels/`: Generated runtime artifacts.

## Runtime artifacts

Training currently writes these files into `runtimeModels/`:

- `intent_model.keras`
- `words.pkl`
- `classes.pkl`

The runtime loads these artifacts when the chatbot starts, so they need to exist before running `chatbotChatter.py`.

## Content formats

### Intent files (`runtimeInfo/Intents/*.yaml`)

Required keys:

- `name`: intent name
- `patterns`: example user utterances
- `responses`: canned replies

Example:

```yaml
name: quiz
patterns:
  - "Start quiz"
responses:
  - "Sure, I can help you start a quiz."
```

### Quiz files (`runtimeInfo/quiz/*.yaml`)

Common keys:

- `name`
- `description`
- `questions`: list of question objects

Question fields commonly include:

- `question`: prompt text
- `type`: fill-in-the-blank or multiple-choice
- `answer`: accepted answer(s) or option number
- `options`: required for multiple-choice questions
- `capitalization`: regard or disregard for fill-in-the-blank answers
- `reason`: explanation shown when the answer is wrong

### User files (`runtimeInfo/userInfo/*.yaml`)

Common keys:

- `username`
- `salt`
- `hashed_password`
- `quiz_progress`

## Setup

### Prerequisites

- Python 3.10+ recommended
- `pip`

Install the core dependencies from this folder:

```bash
pip install tensorflow keras nltk numpy pyyaml matplotlib
```

Notes:

- The runtime and training scripts call `nltk.download(...)` automatically.
- If the machine is offline or restricted, install the required NLTK data ahead of time.

## Train the intent model

Run this from the `Support Chatbot Server` folder:

```bash
python chatbotIntentTrainer.py
```

This refreshes the files in `runtimeModels/`.

## Start the chatbot

Run:

```bash
python chatbotChatter.py
```

Expected flow:

1. The bot asks for username.
2. The bot asks for password.
3. On success, the bot hands off to the welcome/menu handler.
4. The user can choose encouragement, quiz, or chat.

## Start the Telegram bot

Install dependency:

```bash
pip install python-telegram-bot
```

Set token in environment variable or `.env`:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

Optional timeout setting:

```env
TELEGRAM_IDLE_TIMEOUT_SECONDS=600
```

Run:

```bash
python chatbotHeadlessTelegramBot.py
```

Telegram runtime notes:

- Uses the same plugin state machine (`handler`, `state`, `meta`) as `chatbotChatter.py`.
- Maintains isolated per-chat sessions by Telegram `chat_id`.
- Supports `/start` and `/reset` commands.
- Sends typing indicators while slower responses are being generated.
- Auto-resets idle sessions after `TELEGRAM_IDLE_TIMEOUT_SECONDS`.
- Writes activity logs to `headlessLog/telegram_YYYY-MM-DD.jsonl`.

Log fields currently include:

- `timestamp_utc`
- `chat_id`
- `user_id`
- `direction` (`received` or `sent`)
- `text`

Security note:

- Password input received in `LoginHandler` password state is logged as `<masked_password>`.

## Flow handoff

Each handler returns a dictionary with:

- `response`
- `next_handler`
- `next_state`
- `meta_update`

The runtime loop applies those values to the current conversation state. If a handoff state is returned, the next handler is dispatched automatically so the user does not need to restart the program.

## Extending the project

### Add a new flow plugin

1. Create a new file under `runtimeFlowPlugins/`.
2. Register a function with `@runtimeFlowPlugins.register("YourHandlerName")`.
3. Keep the function signature consistent: `(state, meta, inputText, predictedIntent)`.
4. Return the standard outcome dictionary.
5. Route to the new handler from `welcomeHandlers.py` or another handler.

### Add new intents

1. Add a YAML file under `runtimeInfo/Intents/`.
2. Retrain with `chatbotIntentTrainer.py`.
3. Confirm the `runtimeModels/` files are updated.

### Add quiz content

1. Add a new YAML file under `runtimeInfo/quiz/`.
2. Make sure each question includes the required fields.
3. The quiz handler will pick up the new set automatically.

### Add retrieval knowledge

1. Add topic-based `.md`, `.txt`, or `.yaml` files under `runtimeInfo/ragKnowledge/`.
2. Keep the notes clear and keyword-friendly.
3. Restart the chatbot runtime so the retrieval index is rebuilt.

## Notes and limitations

- `chatbotNLP.py` loads the model and pickles on import, so missing runtime artifacts will stop the chatbot from starting.
- Malformed intent YAML raises `ValueError` by design.
- Quiz matching supports exact, numeric, alias-like, and fuzzy matching, but ambiguous text can still select the wrong set.
- The chat flow uses local retrieval from the files in `runtimeInfo/ragKnowledge/` and does not call external APIs.

## Typical workflow

1. Update intents, quiz sets, encouragements, or retrieval notes in `runtimeInfo/`.
2. Retrain the intent model if the intents changed.
3. Run `chatbotChatter.py` and test the full flow: login -> welcome -> feature flow -> return to menu.
4. Check the user YAML files after quizzes to confirm progress was saved.

## Troubleshooting

- Model file missing: run `chatbotIntentTrainer.py` first.
- NLTK resource error: run once with internet access or install `punkt` and `wordnet` manually.
- Login always fails: confirm the user file exists and the password hash is generated with the correct password and salt order.
- New plugin not called: confirm the file is under `runtimeFlowPlugins/` and the handler name matches the handoff target.

## Suggested follow-up

- Pin the dependency versions in `requirements.txt`.
- Remove sensitive debug prints from login.
- Add tests for plugin transitions and quiz evaluation.
- Add schema checks for quiz and user YAML files.
