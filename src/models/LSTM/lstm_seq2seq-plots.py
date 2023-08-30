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
model_name = 'LSTM-Seq2Seq'

for index,val in enumerate(list_of_stocks):

    # Load and preprocess the data
    data = pd.read_csv('data/{}/{}_transformed.csv'.format(list_of_stocks[index], list_of_stocks[index]))
    scaler = MinMaxScaler()
    sequence_length = 5
    features = []
    labels = []
    for i in range(sequence_length, len(data)):
        features.append(data.iloc[i-sequence_length:i, :-1].values)
        labels.append(data.iloc[i-sequence_length:i, -1].values)
    features = np.array(features)
    labels = np.array(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    features_train = scaler.fit_transform(features_train.reshape(-1, features_train.shape[-1])).reshape(features_train.shape)

    # Transform the test data using the fitted scaler
    features_test = scaler.transform(features_test.reshape(-1, features_test.shape[-1])).reshape(features_test.shape)
    features_train = torch.tensor(features_train).float().transpose(0, 1)
    labels_train = torch.tensor(labels_train).long().transpose(0, 1)
    features_test = torch.tensor(features_test).float().transpose(0, 1)
    labels_test = torch.tensor(labels_test).long().transpose(0, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Define the model
    class Encoder(nn.Module):
        def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
            super().__init__()
            
            self.fc = nn.Linear(input_dim, emb_dim)
            self.fc_norm = nn.LayerNorm(emb_dim)  # LayerNorm after fc
            self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
            self.lstm_norm = nn.LayerNorm(hidden_dim)  # LayerNorm after lstm
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, src):
            # src shape: [sequence_length, batch_size]
            
            embedded = self.dropout(self.fc_norm(self.fc(src)))
            # embedded shape: [sequence_length, batch_size, emb_dim]
            
            outputs, (hidden, cell) = self.lstm(embedded)
            hidden = self.lstm_norm(hidden)
            cell = self.lstm_norm(cell)
            return hidden, cell

    class Decoder(nn.Module):
        def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
            super().__init__()
            
            self.output_dim = output_dim
            self.embedding = nn.Embedding(output_dim, emb_dim)
            self.embedding_norm = nn.LayerNorm(emb_dim)  # LayerNorm after embedding
            self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
            self.rnn_norm = nn.LayerNorm(hidden_dim)  # LayerNorm after rnn
            self.fc_out = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, input, hidden, cell):
            # input shape: [batch_size]
            
            input = input.unsqueeze(0)
            # input shape: [1, batch_size]
            
            embedded = self.dropout(self.embedding_norm(self.embedding(input)))
            # embedded shape: [1, batch_size, emb_dim]
            
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            hidden = self.rnn_norm(hidden)
            cell = self.rnn_norm(cell)
            prediction = self.fc_out(output.squeeze(0))
            # prediction shape: [batch_size, output_dim]
            
            return prediction, hidden, cell



    class LSTMSeq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super().__init__()
            
            self.encoder = encoder
            self.decoder = decoder
            self.device = device
            
        def forward(self, src, trg, teacher_forcing_ratio=0.5):
            # src shape: [src_sequence_length, batch_size]
            # trg shape: [trg_sequence_length, batch_size]
            
            trg_len, batch_size = trg.shape
            trg_vocab_size = self.decoder.output_dim
            
            # tensor to store decoder outputs
            outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
            
            # last hidden state of the encoder is the context
            hidden, cell = self.encoder(src)
            
            # first input to the decoder is the <START> token
            input = trg[0,:]
            
            for t in range(1, trg_len):
                # insert input token embedding, previous hidden and cell states
                # receive output tensor (predictions) and new hidden and cell states
                output, hidden, cell = self.decoder(input, hidden, cell)
                
                # save prediction to output tensor
                outputs[t] = output
                
                # determine if we will use teacher forcing or not
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                
                # get the highest predicted token from softmax
                top1 = output.argmax(1) 
                
                # if teacher forcing, use actual next token as next input. If not, use predicted token
                input = trg[t] if teacher_force else top1
            
            return outputs


    # Parameters for the new LSTM model
    input_dim = 22
    output_dim = 2
    hidden_dim = 64
    num_layers = 1

    encoder = Encoder(input_dim, hidden_dim, hidden_dim, num_layers, dropout=0.2)
    decoder = Decoder(output_dim, hidden_dim, hidden_dim, num_layers, dropout=0.2)
    model = LSTMSeq2Seq(encoder, decoder, device)

    counts = pd.Series(labels_train.reshape(-1).numpy()).value_counts().values
    class_weights = torch.tensor(counts.sum() / counts).float()
    class_weights = class_weights.flip(dims=[0])
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

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
                dummy = torch.full((1, labels.shape[1]), 0, dtype=torch.long)
                decoder_input = torch.cat([dummy, labels[:-1]], dim=0)
                outputs = model(features, decoder_input)
                outputs = outputs.view(-1, outputs.shape[-1])  # reshape for loss computation
                labels = labels.view(-1)  # reshape for loss computation
                
                # Mask out dummy labels
                mask = labels != -1
                loss = loss_fn(outputs[mask], labels[mask])
                val_loss += loss.item() * mask.sum().item()  # accumulate the weighted loss
                
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
                # Teacher forcing: use the true labels as input to the decoder
                dummy = torch.full((1, labels.shape[1]), 0, dtype=torch.long)
                decoder_input = torch.cat([dummy, labels[:-1]], dim=0)

                # Forward pass
                outputs = model(features, decoder_input)
                outputs = outputs.view(-1, outputs.shape[-1])  # reshape for loss computation
                labels = labels.view(-1)  # reshape for loss computation

                # Compute loss only for non-dummy labels
                mask = labels != -1
                loss = loss_fn(outputs[mask], labels[mask])
                total_loss += loss.item() * mask.sum().item()  # accumulate the total loss

                # Calculate the number of correct predictions
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += mask.sum().item()

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
    train_data = TensorDataset(features_train, labels_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data = TensorDataset(features_test, labels_test)
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
