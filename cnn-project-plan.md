# CNN Image Classification - Team Plan (G3: Alejandro, Baggiyam, Michael)

## Shared Foundation (All Members)
- **Data Exploration & Preprocessing**: Create unified, reusable preprocessing pipeline
  - [x] Load and explore dataset together
  - [ ] Train initial model together (for familiarity)
  - [ ] Evaluate the model
  - [ ] Design standardized data transformations (normalization, resizing)
  - [ ] Create data augmentation strategies
  - [ ] Package in `data_utils.py` module for consistent use across experiments

- **Evaluation Framework**: Develop standard evaluation utilities
  - Implement consistent metrics calculation (accuracy, precision, recall, F1)
  - Create standardized visualization functions (confusion matrices, history plots)
  - Build model comparison tools for fair evaluation across experiments
  - Package in `evaluation_utils.py` module for team-wide use

- **Base Model Architecture**: Collaboratively design modular CNN framework
  - Create `model_utils.py` with building blocks for CNN architecture:
    - `create_conv_block()`: Configurable convolutional block (conv+pooling+normalization+dropout)
    - `create_classifier_head()`: Configurable dense classification layers
    - `create_base_model()`: Skeleton architecture everyone can extend
  - Document architecture decisions and design patterns
  - Implement version control for model variations

