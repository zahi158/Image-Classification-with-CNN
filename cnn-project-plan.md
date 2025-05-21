# CNN Image Classification - Team Plan (G3: Alejandro, Baggiyam, Michael)

## Shared Foundation (All Members)
- **Data Exploration & Preprocessing**: Create unified, reusable preprocessing pipeline
  - [ ] Load and explore dataset together
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

## Individual Responsibilities

### Person 1: CNN Architecture Exploration
- Focus on **depth and width variations**:
  - Experiment with deeper networks (more layers)
  - Test wider networks (more filters per layer)
  - Implement residual connections
  - Create subclass `DeepCNN` extending base architecture

### Person 2: Regularization & Optimization Strategies
- Focus on **training dynamics**:
  - Implement advanced regularization (L1/L2, dropout schemes)
  - Test learning rate schedules and adaptive optimizers
  - Experiment with batch normalization strategies
  - Create subclass `RegularizedCNN` extending base architecture

### Person 3: Transfer Learning & Deployment
- Focus on **pre-trained models and features**:
  - Implement feature extraction from various backbones
  - Experiment with fine-tuning strategies
  - Develop mixed models (custom + pre-trained features)
  - Create subclass `TransferCNN` extending base architecture
  - Build Flask deployment for best model

## Model Variation Framework
```python
# Example of model architecture variations sharing common code

# Base model template in model_utils.py
def create_base_model(input_shape, num_classes):
    """Base CNN architecture that all team members extend"""
    # Common model structure everyone uses
    inputs = tf.keras.Input(shape=input_shape)
    # Initial convolutional block
    x = create_conv_block(inputs, filters=32, kernel_size=3)
    # Additional blocks to be customized by each person
    # ...
    # Final classification head
    outputs = create_classifier_head(x, num_classes)
    model = tf.keras.Model(inputs, outputs)
    return model

# Person 1's variation (model_depth.py)
class DeepCNN(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Reuse base architecture but add depth
        self.base_layers = get_base_layers(input_shape)
        # Add extra layers specific to this architecture
        self.extra_conv1 = create_conv_block(filters=64, kernel_size=3)
        self.extra_conv2 = create_conv_block(filters=128, kernel_size=3)
        # Shared classifier head
        self.classifier = create_classifier_head(num_classes)
    
    def call(self, inputs):
        x = self.base_layers(inputs)
        x = self.extra_conv1(x)
        x = self.extra_conv2(x)
        return self.classifier(x)

# Similar patterns for Person 2 and Person 3's models
```

## Collaboration Process for Model Development
1. **Design base architecture together** (Week 1)
   - Jointly create `model_utils.py` with shared components
   - Agree on model interface and evaluation metrics
   - Set up experiment tracking system

2. **Parallel development with daily check-ins** (Week 2)
   - Each person develops their model variation
   - Share insights and results in daily standup
   - Maintain shared evaluation criteria

3. **Cross-pollination of ideas** (Week 3)
   - Identify best features from each approach
   - Create combined model incorporating best elements
   - Joint review of all models

4. **Final model selection and deployment** (Week 3-4)
   - Select best-performing architecture
   - Optimize final model
   - Deploy and document

## Technical Implementation
- **data_utils.py**: Shared data processing
- **evaluation_utils.py**: Shared evaluation
- **model_utils.py**: Shared model components
- **model_depth.py**: Person 1's architecture 
- **model_regularization.py**: Person 2's architecture
- **model_transfer.py**: Person 3's architecture
- **ensemble.py**: Combined model leveraging best approaches

## Success Metrics
- Base CNN variations: 70-80% accuracy
- Transfer learning variations: 85-90% accuracy
- Final ensemble model: >90% accuracy
- Working Flask app for prediction
