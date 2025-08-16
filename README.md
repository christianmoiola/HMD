# CarLLaMA: Your AI Car Dealership Assistant

CarLLaMA is a conversational AI assistant designed to streamline the car buying process. Powered by Hugging Face's LLaMA models, this assistant can help users with everything from getting car information and negotiating prices to booking appointments and purchasing a vehicle. The system uses a modular pipeline with distinct components for natural language understanding (NLU), dialog management (DM), and natural language generation (NLG) to provide a seamless multi-turn conversational experience.

-----

## Key Features

  * **Multi-turn Conversations**: Handles complex, ongoing conversations about buying a car.
  * **Customizable Prompts**: Fine-tune the assistant's behavior for each component (PRE\_NLU, NLU, DM, NLG) with customizable prompts.
  * **Flexible Models**: Supports popular Hugging Face models, including `Meta-LLaMA-3-8B-Instruct` and `LLaMA-2-7b-chat-hf`.
  * **Effortless Integration**: Easily integrate your own car information datasets in JSON format.
  * **Evaluation Pipeline**: Test and improve the performance of your NLU and PRE\_NLU components.

-----

## Installation

### 1\. Clone the Repository

Get started by cloning the project from GitHub.

```bash
git clone https://github.com/christianmoiola/HMD.git
cd HMD
```

### 2\. Install Dependencies

Install all the necessary Python libraries with a single command.

```bash
pip install -r requirements.txt
```

-----

## Hugging Face Token Setup

To access the powerful Hugging Face models, you'll need to set up your access token.

1.  **Get Your Token**: If you don't have an account, register on the [Hugging Face website](https://huggingface.co). Once logged in, go to your settings to generate a new access token.
2.  **Configure the File**: Rename `token_example.ini` to **`token.ini`**. Open the file and paste your token in the designated field:

<!-- end list -->

```ini
[HUGGING_FACE]
token=your_hf_token_here
```

This token enables the project to download and utilize the LLaMA models.

-----

## Configuration (`config.ini`)

The `config.ini` file is your control panel for the entire system. Adjust these parameters to customize CarLLaMA's behavior.

### General Settings

  * **`folder_model`**: The directory where models are stored. The default is `models`.
  * **`model_name`**: Choose your preferred model. Use `llama3` for `Meta-LLaMA-3-8B-Instruct` or `llama2` for `LLaMA-2-7b-chat-hf`.
  * **`dtype`**: The data type for model computations. `bf16` is the default.
  * **`max_seq_length`**: Sets the maximum number of tokens the model can handle in a sequence.
  * **`initial_message`**: The welcoming message the assistant sends to users at the start of a conversation.

### Prompts per Component

You can specify the prompt for each component by editing the paths in `config.ini`. You have the flexibility to choose between two prompt folders: `prompts_base` for prompts without examples or `prompts_examples`

  * **PRE\_NLU**: The prompt used before the main NLU stage.
  * **NLU**: Prompts are set for specific intents, such as `prompt_buying_car`.
  * **DM**: The prompt that defines the dialog manager's logic.
  * **NLG**: Prompts for generating responses based on different intents, like `prompt_order_car`.

### Database

  * **`path`**: The file path to your JSON dataset containing car information, for example, `dataset/car_dataset.json`.

-----

## Running the Assistant

Follow these simple steps to start CarLLaMA.

1.  Ensure your `config.ini` and `token.ini` files are correctly configured.
2.  Run the main script from your terminal:

<!-- end list -->

```bash
python main.py
```

You can easily change the model, initial message, and prompts by simply editing the `config.ini` file.

-----

## 📊 Evaluation

CarLLaMA includes a dedicated pipeline for evaluating individual components.

  * **NLU**: Test cases are located in `src/evaluation/data/nlu_evaluation.json`.
  * **PRE\_NLU**: Test cases are located in `src/evaluation/data/pre_nlu_evaluation.json`.

To evaluate only specific components, you can comment out the execution of other parts of the pipeline in `main.py`.