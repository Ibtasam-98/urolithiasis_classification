import multiprocessing as mp
from config import config
from utils import set_seeds, save_models, print_final_comparison
from data_loader import load_datasets, preprocess_datasets
from models import create_dnn_model, create_mlp_model, create_autoencoder_dnn_model
from evaluation import evaluate_model
from visualization import (plot_unified_training_history, create_comparison_visualizations,
                       plot_epoch_comparison, create_model_architecture_diagrams)

def main():
    # Set seeds for reproducibility
    set_seeds()

    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_ds, val_ds, test_ds, class_names = load_datasets()
    train_ds_flat, val_ds_flat, test_ds_flat, INPUT_DIM = preprocess_datasets(train_ds, val_ds, test_ds)

    # Initialize dictionaries to store models and results
    models_dict = {}
    histories = []
    results_dict = {}
    model_names = ["DNN", "MLP", "AE-DNN"]

    # Train and evaluate DNN model
    print("\n" + "=" * 60)
    print("TRAINING DEEP NEURAL NETWORK (DNN)")
    print("=" * 60)
    dnn = create_dnn_model(INPUT_DIM)
    hist_dnn = dnn.fit(train_ds_flat, validation_data=val_ds_flat, epochs=10, verbose=1)
    res_dnn = evaluate_model(dnn, train_ds_flat, val_ds_flat, test_ds_flat, "kidney_stone_dnn", class_names, hist_dnn)
    models_dict["DNN"] = dnn
    histories.append(hist_dnn.history)
    results_dict["DNN"] = res_dnn

    # Train and evaluate MLP model
    print("\n" + "=" * 60)
    print("TRAINING SMALLER MLP")
    print("=" * 60)
    mlp = create_mlp_model(INPUT_DIM)
    hist_mlp = mlp.fit(train_ds_flat, validation_data=val_ds_flat, epochs=10, verbose=1)
    res_mlp = evaluate_model(mlp, train_ds_flat, val_ds_flat, test_ds_flat, "kidney_stone_mlp", class_names, hist_mlp)
    models_dict["MLP"] = mlp
    histories.append(hist_mlp.history)
    results_dict["MLP"] = res_mlp

    # Train and evaluate Autoencoder-based DNN model
    print("\n" + "=" * 60)
    print("TRAINING AUTOENCODER-BASED DNN CLASSIFIER")
    print("=" * 60)
    autoencoder, classifier = create_autoencoder_dnn_model(INPUT_DIM)
    autoencoder.fit(train_ds_flat.map(lambda x, y: (x, x)),
                    validation_data=val_ds_flat.map(lambda x, y: (x, x)),
                    epochs=10, verbose=1)
    hist_cls = classifier.fit(train_ds_flat, validation_data=val_ds_flat, epochs=10, verbose=1)
    res_ae = evaluate_model(classifier, train_ds_flat, val_ds_flat, test_ds_flat, "kidney_stone_autoencoder_dnn",
                            class_names, hist_cls)
    models_dict["AE-DNN"] = classifier
    histories.append(hist_cls.history)
    results_dict["AE-DNN"] = res_ae

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_unified_training_history(histories, model_names, config.SAVE_FIGS_DIR)
    create_comparison_visualizations(results_dict, config.SAVE_FIGS_DIR)
    plot_epoch_comparison(histories, model_names, config.SAVE_FIGS_DIR)
    create_model_architecture_diagrams(models_dict, config.SAVE_FIGS_DIR)

    # Save models
    print("\nSaving models...")
    save_models(models_dict)

    # Print final comparison
    print_final_comparison(results_dict, model_names)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()