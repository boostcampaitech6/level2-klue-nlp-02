from transformers import AutoConfig, AutoModelForSequenceClassification


def get_model(MODEL_NAME, device):
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    model.parameters
    model.to(device)

    return model
