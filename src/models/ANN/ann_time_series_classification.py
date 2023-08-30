
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import os

#Helper function used to write final result to text file
def write_or_append_to_file(path, content):
    # Ensure directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write or append to file
    with open(path, 'a') as f:
        f.write(content)

# List of stocks to train and evaluate, top ten bby market cap within the NASDAQ
list_of_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'ASML', 'AVGO', 'PEP']
model_name = 'ANN'

for i, val in enumerate(list_of_stocks):
    # Load and preprocess the data
    data = pd.read_csv('data/{}/{}_transformed.csv'.format(list_of_stocks[i], list_of_stocks[i]))
    scaler = MinMaxScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    # For ANN, use single time step data
    features = data.iloc[:-1, :-1].values
    labels = data.iloc[1:, -1].values
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    features_train = torch.tensor(features_train).float()
    labels_train = torch.tensor(labels_train).long()
    features_test = torch.tensor(features_test).float()
    labels_test = torch.tensor(labels_test).long()

    class SimpleANN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
            super(SimpleANN, self).__init__()
            
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)  
            self.dropout1 = nn.Dropout(dropout_prob)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)  
            self.dropout2 = nn.Dropout(dropout_prob)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.softmax(self.fc3(x))
            return x

 

    # Assuming the number of features in the data is 'n_features' and number of classes is 'n_classes'
    model = SimpleANN(features_train.shape[1], 64, len(torch.unique(labels_train)), dropout_prob=0.2)

    # Consider the class imbalance
    counts = pd.Series(labels_train.reshape(-1).numpy()).value_counts().values
    class_weights = torch.tensor(counts.sum() / counts).float()
    class_weights = class_weights.flip(dims=[0])
    # Loss and Optimizer
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # DataLoader for batch processing
    batch_size = 32
    train_data = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data = TensorDataset(features_test, labels_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)


    # Helper function to calculate FPR and FDR
    def calculate_fpr_fdr(y_true, y_pred):
        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate the False Positive Rate (FPR)
        fpr = fp / (fp + tn)

        # Calculate the False Discovery Rate (FDR)
        fdr = fp / (fp + tp)

        return fpr, fdr

    # Helper function to evaluate the model (Validation)
    def evaluate(model, test_loader, loss_fn):
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_predicted = []

        with torch.no_grad():
            for features, labels in test_loader:
                outputs = model(features)
                outputs = outputs.view(-1, outputs.shape[-1])  # reshape for loss computation
                labels = labels.view(-1)  # reshape for loss computation
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()  # accumulate validation loss
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_predicted.extend(predicted.numpy())

        val_loss /= len(all_labels)  # average out the validation loss
        
        total_correct = (torch.tensor(all_predicted) == torch.tensor(all_labels)).sum().item()
        total_samples = len(all_labels)
        accuracy = total_correct / total_samples

        report = classification_report(all_labels, all_predicted, target_names=['0', '1'])
        fpr, fdr = calculate_fpr_fdr(all_labels, all_predicted)
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predicted, labels=[0, 1])
        
        return report, fpr, fdr, accuracy, precision, recall, f1_score, val_loss

    # Train and evaluate the model
    def train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, epochs, stock_name):
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_fprs = []
        train_fdrs = []
        eval_fprs = []
        eval_fdrs = []
        train_precisions = []
        train_recalls = []
        train_f1_scores = []
        val_precisions = []
        val_recalls = []
        val_f1_scores = []
        
        for epoch in range(epochs):
        
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            train_labels = []
            train_predicted = []

            for i, (features, labels) in enumerate(train_loader):
            
                # Forward pass
                outputs = model(features)
                outputs = outputs.view(-1, outputs.shape[-1])  # reshape for loss computation
                labels = labels.view(-1)  # reshape for loss computation

                # Compute loss only for non-dummy labels
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() # accumulate the total loss

                # Calculate the number of correct predictions
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Accumulate labels and predictions for calculating FPR and FDR for training data
                train_labels.extend(labels.numpy())
                train_predicted.extend(predicted.numpy())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples
            train_fpr, train_fdr = calculate_fpr_fdr(train_labels, train_predicted)
            precision, recall, f1_score, _ = precision_recall_fscore_support(train_labels, train_predicted, labels=[0, 1])

            
            # Evaluate the model
            report, fpr, fdr, val_accuracy, val_precision, val_recall, val_f1_score, val_loss = evaluate(model, test_loader, loss_fn)
            print(f'Epoch {epoch+1}/{epochs},Evaluation report:\n{report}')
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Accuracy: {accuracy}')
            print(f'Training FPR: {train_fpr}, Training FDR: {train_fdr}')
            print(f'Evaluation FPR: {fpr}, Evaluation FDR (How often we lose money): {fdr}')
            print(f'Validation Accuracy: {val_accuracy}')
            train_losses.append(avg_loss)
            val_losses.append(val_loss)
            train_accuracies.append(accuracy)
            val_accuracies.append(val_accuracy)
            train_fprs.append(train_fpr)
            train_fdrs.append(train_fdr)
            eval_fprs.append(fpr)
            eval_fdrs.append(fdr)
            # Append precision, recall and F1 score to lists
            train_precisions.append(precision)
            train_recalls.append(recall)
            train_f1_scores.append(f1_score)
            
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_f1_scores.append(val_f1_score)
        
        #Save last epoch checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, 'results/checkpoints/ANN/{}-checkpoint.pth'.format(stock_name))
        #Save Final Metrics in file
        content = (
            f'Epoch {epoch+1}/{epochs}, Evaluation report:\n{report}\n'
            f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Accuracy: {accuracy}\n'
            f'Training FPR: {train_fpr}, Training FDR: {train_fdr}\n'
            f'Evaluation FPR: {fpr}, Evaluation FDR (How often we lose money): {fdr}\n'
            f'Validation Accuracy: {val_accuracy}\n'
        )
        write_or_append_to_file("results/final_metrics/ANN/{}.txt".format(stock_name), content)
        return train_losses, train_accuracies, val_accuracies, train_fprs, train_fdrs, eval_fprs, eval_fdrs, train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores, val_losses


    # Choose the Batch Size, and transform the data into DataLoader objects
    batch_size = 32
    train_data = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data = TensorDataset(features_test, labels_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    # Choose Epochs, and train the model
    epochs=2500
    train_losses, train_accuracies, val_accuracies, train_fprs, train_fdrs, eval_fprs, eval_fdrs, train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores, val_losses=train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, epochs=epochs, stock_name=list_of_stocks[i])
    # Plotting metrics after training
    epochs_range = range(1, epochs + 1)

    # Plot all the results 
    # Printing lengths of metric lists before plotting
    print(train_losses)

    plt.figure(figsize=(14, 5))

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/figures/{}/{}_plots1.png'.format(model_name, list_of_stocks[i]))
    plt.close()


    # Plotting FPR and FDR
    plt.figure(figsize=(14, 5))

    # Plotting training FPR
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_fprs, label='Training FPR')
    plt.plot(epochs_range, eval_fprs, label='Validation FPR')
    plt.title('Training and Validation FPR')
    plt.xlabel('Epochs')
    plt.ylabel('FPR')
    plt.legend()

    # Plotting training and validation FDR
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_fdrs, label='Training FDR')
    plt.plot(epochs_range, eval_fdrs, label='Validation FDR')
    plt.title('Training and Validation FDR')
    plt.xlabel('Epochs')
    plt.ylabel('FDR')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/figures/{}/{}_plots2.png'.format(model_name, list_of_stocks[i]))
    plt.close()

    plt.figure(figsize=(15, 10))

    # Plotting precision
    plt.subplot(3, 2, 1)
    plt.plot(range(epochs), [p[0] for p in train_precisions], label='Train Precision (Class 0)', color='blue')
    plt.plot(range(epochs), [p[1] for p in train_precisions], label='Train Precision (Class 1)', color='blue', linestyle='dashed')
    plt.plot(range(epochs), [p[0] for p in val_precisions], label='Validation Precision (Class 0)', color='red')
    plt.plot(range(epochs), [p[1] for p in val_precisions], label='Validation Precision (Class 1)', color='red', linestyle='dashed')
    plt.title('Precision over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plotting recall
    plt.subplot(3, 2, 2)
    plt.plot(range(epochs), [r[0] for r in train_recalls], label='Train Recall (Class 0)', color='blue')
    plt.plot(range(epochs), [r[1] for r in train_recalls], label='Train Recall (Class 1)', color='blue', linestyle='dashed')
    plt.plot(range(epochs), [r[0] for r in val_recalls], label='Validation Recall (Class 0)', color='red')
    plt.plot(range(epochs), [r[1] for r in val_recalls], label='Validation Recall (Class 1)', color='red', linestyle='dashed')
    plt.title('Recall over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Plotting F1 Score
    plt.subplot(3, 2, 3)
    plt.plot(range(epochs), [f[0] for f in train_f1_scores], label='Train F1 Score (Class 0)', color='blue')
    plt.plot(range(epochs), [f[1] for f in train_f1_scores], label='Train F1 Score (Class 1)', color='blue', linestyle='dashed')
    plt.plot(range(epochs), [f[0] for f in val_f1_scores], label='Validation F1 Score (Class 0)', color='red')
    plt.plot(range(epochs), [f[1] for f in val_f1_scores], label='Validation F1 Score (Class 1)', color='red', linestyle='dashed')
    plt.title('F1 Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/figures/{}/{}_plots3.png'.format(model_name, list_of_stocks[i]))
    plt.close()
