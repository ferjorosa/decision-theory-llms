---
title: Oil Field Purchase Decision
emoji: 😻
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
short_description: Oil field purchase decision problem
---

# Oil Field Purchase Decision Analysis

This Gradio app provides a decision support tool for analyzing oil field purchase decisions using influence diagrams and PyAgrum.

## Deployment to Hugging Face Spaces

To update this Space from the main repository:

> **Note**: We use `git subtree` because this Gradio app lives in a subdirectory of the main repository, while the Hugging Face Space has its own separate git history. The subtree command allows us to push only this specific directory to the Space.

1. **Set up authentication** (one-time setup):
   ```bash
   # Add your HUGGINGFACE_TOKEN to .env file in project root
   echo "HUGGINGFACE_TOKEN=your_token_here" >> .env
   
   # Add HF Space as remote
   git remote add hf https://ferjorosa:${HUGGINGFACE_TOKEN}@huggingface.co/spaces/ferjorosa/oil-field-purchase-decision
   ```

2. **Deploy updates**:
   ```bash
   # Load environment variables
   source .env
   
   # Push gradio app directory to HF Space
   git push hf `git subtree split --prefix=decision_problems/oil_field_purchase/gradio_app`:main --force
   ```

## Configuration Reference
Check out the [Hugging Face Spaces configuration reference](https://huggingface.co/docs/hub/spaces-config-reference)
