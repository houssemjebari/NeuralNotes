
# Reinforcement Learning from Human Feedback (RLHF) Course

This repository contains the notebooks and materials from the AI RLHF Course on DeepLearning.AI. The course covers key aspects of Reinforcement Learning from Human Feedback (RLHF), including dataset exploration, tuning a large language model (LLM), and evaluating the tuned model.

Certificate: [View My Certificate](https://learn.deeplearning.ai/accomplishments/eb917f2e-24e5-4029-b86e-7e252da56518)

## Course Contents

### Explore Data (`Explore Data.ipynb`)
- Understanding datasets required for RLHF:
  - Preference dataset: Contains input prompts, candidate responses, and human preferences.
  - Prompt dataset: Contains only input prompts.
- Loading and analyzing dataset files in `.jsonl` format.
- Preparing data for the RLHF pipeline.

### Tune LLM (`Tune LLM.ipynb`)
- Fine-tuning an LLM using RLHF.
- Using Google Cloud Pipeline Components for RLHF training.
- Running on KubeFlow Pipelines locally or in the cloud.
- Installing dependencies and compiling the RLHF pipeline.

### Evaluate Model (`Evaluate Model.ipynb`)
- Evaluating the tuned LLM.
- Using TensorBoard to visualize training results.
- Setting up logging directories for reward modeling.
