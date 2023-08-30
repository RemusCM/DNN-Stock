from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import os

def write_or_append_to_file(path, content):
    # Ensure directory exists
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write or append to file
    with open(path, 'a') as f:
        f.write(content)

list_of_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'ASML', 'AVGO', 'PEP']
model_name = 'LSTM-One2One'

for index,val in enumerate(list_of_stocks):

    # Load and preprocess the data
    data = pd.read_csv('data/{}/{}_transformed.csv'.format(list_of_stocks[index], list_of_stocks[index]))

    feature_columns = data.columns[:-1]
    target_column = data.columns[-1]

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data[feature_columns], data[target_column], test_size=0.2, random_state=42)

    # Scale the features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Reshape the data to 3D for LSTM 
    train_data_reshaped = train_data_scaled.reshape((train_data_scaled.shape[0], 1, train_data_scaled.shape[1]))
    test_data_reshaped = test_data_scaled.reshape((test_data_scaled.shape[0], 1, test_data_scaled.shape[1]))

    # Convert the labels to tensors
    labels_train = train_labels.values
    labels_test = test_labels.values
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
            super(LSTM, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.ln2 = nn.LayerNorm(hidden_size)

            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, output_size)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            x = x.float()
            h0 = h0.float()
            c0 = c0.float()
            out, _ = self.lstm(x, (h0, c0))
            out = self.ln2(out[:, -1, :])

            out = self.fc(out)
            return out


    model = LSTM(input_size=22, hidden_size=32, num_layers=2, output_size=2, dropout=0.2).to(device)

    counts = pd.Series(labels_train.reshape(-1)).value_counts().values
    class_weights = torch.tensor(counts.sum() / counts).float()
    class_weights = class_weights.flip(dims=[0])

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the evaluation function
    def calculate_fpr_fdr(y_true, y_pred):
        # Calculate the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate the False Positive Rate (FPR)
        fpr = fp / (fp + tn)

        # Calculate the False Discovery Rate (FDR)
        fdr = fp / (fp + tp)

        return fpr, fdr

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
                labels = labels.long()
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
                labels = labels.view(-1).long()  # reshape for loss computation

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
            print(f'Epoch {epoch+1}/{epochs}')
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
        torch.save(checkpoint, 'results/checkpoints/{}/{}-checkpoint.pth'.format(model_name,stock_name))
        #Save Final Metrics in file
        content = (
            f'Epoch {epoch+1}/{epochs}, Evaluation report:\n{report}\n'
            f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Accuracy: {accuracy}\n'
            f'Training FPR: {train_fpr}, Training FDR: {train_fdr}\n'
            f'Evaluation FPR: {fpr}, Evaluation FDR (How often we lose money): {fdr}\n'
            f'Validation Accuracy: {val_accuracy}\n'
        )
        write_or_append_to_file("results/final_metrics/{}/{}.txt".format(model_name, stock_name), content)
            
        return train_losses, train_accuracies, val_accuracies, train_fprs, train_fdrs, eval_fprs, eval_fdrs, train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores, val_losses


    # Train and evaluate the model
    batch_size = 32
    train_data_reshaped = torch.tensor(train_data_reshaped)
    labels_train = torch.tensor(labels_train)
    test_data_reshaped = torch.tensor(test_data_reshaped)
    labels_test = torch.tensor(labels_test)
    train_data = TensorDataset(train_data_reshaped, labels_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data = TensorDataset(test_data_reshaped, labels_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    epochs=2000
    train_losses, train_accuracies, val_accuracies, train_fprs, train_fdrs, eval_fprs, eval_fdrs, train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores, val_losses=train_and_evaluate(model, train_loader, test_loader, loss_fn, optimizer, epochs=epochs, stock_name=list_of_stocks[index])
    # Plotting metrics after training
    epochs_range = range(1, epochs + 1)

    # Printing lengths of metric lists before plotting
    print(train_losses)
    print("Length of train_losses:", len(train_losses))
    print("Length of train_accuracies:", len(train_accuracies))
    print("Length of val_accuracies:", len(val_accuracies))
    print("Length of train_fprs:", len(train_fprs))
    print("Length of train_fdrs:", len(train_fdrs))
    print("Length of eval_fprs:", len(eval_fprs))
    print("Length of eval_fdrs:", len(eval_fdrs))

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
    plt.savefig('results/figures/{}/{}_plots1.png'.format(model_name, list_of_stocks[index]))
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
    plt.savefig('results/figures/{}/{}_plots2.png'.format(model_name, list_of_stocks[index]))
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
    plt.savefig('results/figures/{}/{}_plots3.png'.format(model_name, list_of_stocks[index]))
    plt.close()