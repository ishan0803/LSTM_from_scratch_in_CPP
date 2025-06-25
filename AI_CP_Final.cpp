#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <random>
#include <iomanip>

using namespace std;

// Hyperparameters
const int INPUT_SIZE = 1;      // Stock price input (1 feature)
const int HIDDEN_SIZE = 32;    // Hidden layer size
const int OUTPUT_SIZE = 1;     // Predict 1 value
const int SEQ_LENGTH = 10;     // Sequence length
const float LEARNING_RATE = 0.001f; // Learning rate
const float GRADIENT_CLIP = 5.0f;  // Gradient clipping threshold

// Random number generator
mt19937 gen(42); // Fixed seed for reproducibility
uniform_real_distribution<double> dist(-0.1, 0.1);

// Activation Functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double tanh_custom(double x) {
    return tanh(x);
}

double sigmoid_derivative(double s) {
    return s * (1 - s);
}

double tanh_derivative(double t) {
    return 1 - t * t;
}

// LSTM Cell
struct LSTMCell {
    vector<vector<double>> Wf, Wi, Wc, Wo; // Weight matrices
    vector<double> bf, bi, bc, bo;         // Bias vectors
    
    // Storage for backward pass
    vector<vector<double>> x_inputs;
    vector<vector<double>> h_states;
    vector<vector<double>> c_states;
    vector<vector<double>> f_gates;
    vector<vector<double>> i_gates;
    vector<vector<double>> c_gates;
    vector<vector<double>> o_gates;

    LSTMCell(int input_size, int hidden_size) {
        // Initialize weights with small random values
        Wf = vector<vector<double>>(hidden_size, vector<double>(input_size + hidden_size));
        Wi = vector<vector<double>>(hidden_size, vector<double>(input_size + hidden_size));
        Wc = vector<vector<double>>(hidden_size, vector<double>(input_size + hidden_size));
        Wo = vector<vector<double>>(hidden_size, vector<double>(input_size + hidden_size));

        for (auto& row : Wf) for (double& val : row) val = dist(gen);
        for (auto& row : Wi) for (double& val : row) val = dist(gen);
        for (auto& row : Wc) for (double& val : row) val = dist(gen);
        for (auto& row : Wo) for (double& val : row) val = dist(gen);

        // Initialize biases
        bf = vector<double>(hidden_size, 0.1);
        bi = vector<double>(hidden_size, 0.1);
        bc = vector<double>(hidden_size, 0.1);
        bo = vector<double>(hidden_size, 0.1);
        
        // Initialize storage for backward pass
        x_inputs.clear();
        h_states.clear();
        c_states.clear();
        f_gates.clear();
        i_gates.clear();
        c_gates.clear();
        o_gates.clear();
        
        // Add initial states
        h_states.push_back(vector<double>(hidden_size, 0));
        c_states.push_back(vector<double>(hidden_size, 0));
    }

    void reset_states() {
        x_inputs.clear();
        h_states.clear();
        c_states.clear();
        f_gates.clear();
        i_gates.clear();
        c_gates.clear();
        o_gates.clear();
        
        h_states.push_back(vector<double>(HIDDEN_SIZE, 0));
        c_states.push_back(vector<double>(HIDDEN_SIZE, 0));
    }

    vector<double> forward(const vector<double>& sequence) {
        // Process each time step in the sequence
        for (size_t t = 0; t < sequence.size(); t++) {
            vector<double> x = {sequence[t]};
            x_inputs.push_back(x);
            
            vector<double> prev_h = h_states.back();
            vector<double> prev_c = c_states.back();
            
            // Concatenate previous hidden state and input
            vector<double> concat;
            concat.insert(concat.end(), prev_h.begin(), prev_h.end());
            concat.insert(concat.end(), x.begin(), x.end());
            
            // Calculate gates
            vector<double> f_gate(HIDDEN_SIZE);
            vector<double> i_gate(HIDDEN_SIZE);
            vector<double> c_gate(HIDDEN_SIZE);
            vector<double> o_gate(HIDDEN_SIZE);
            
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                double sum_f = bf[i], sum_i = bi[i], sum_c = bc[i], sum_o = bo[i];
                for (size_t j = 0; j < concat.size(); j++) {
                    sum_f += Wf[i][j] * concat[j];
                    sum_i += Wi[i][j] * concat[j];
                    sum_c += Wc[i][j] * concat[j];
                    sum_o += Wo[i][j] * concat[j];
                }
                f_gate[i] = sigmoid(sum_f);
                i_gate[i] = sigmoid(sum_i);
                c_gate[i] = tanh_custom(sum_c);
                o_gate[i] = sigmoid(sum_o);
            }
            
            // Save gates for backward pass
            f_gates.push_back(f_gate);
            i_gates.push_back(i_gate);
            c_gates.push_back(c_gate);
            o_gates.push_back(o_gate);
            
            // Calculate new cell state and hidden state
            vector<double> new_c(HIDDEN_SIZE);
            vector<double> new_h(HIDDEN_SIZE);
            
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                new_c[i] = f_gate[i] * prev_c[i] + i_gate[i] * c_gate[i];
                new_h[i] = o_gate[i] * tanh_custom(new_c[i]);
            }
            
            c_states.push_back(new_c);
            h_states.push_back(new_h);
        }
        
        return h_states.back(); // Return the final hidden state
    }
    
    void backward(const vector<double>& dh_next, const vector<double>& dc_next, float learning_rate) {
        int sequence_length = x_inputs.size();
        
        vector<double> dh = dh_next;
        vector<double> dc = dc_next;
        
        // Gradients for weights and biases
        vector<vector<double>> dWf(HIDDEN_SIZE, vector<double>(INPUT_SIZE + HIDDEN_SIZE, 0));
        vector<vector<double>> dWi(HIDDEN_SIZE, vector<double>(INPUT_SIZE + HIDDEN_SIZE, 0));
        vector<vector<double>> dWc(HIDDEN_SIZE, vector<double>(INPUT_SIZE + HIDDEN_SIZE, 0));
        vector<vector<double>> dWo(HIDDEN_SIZE, vector<double>(INPUT_SIZE + HIDDEN_SIZE, 0));
        
        vector<double> dbf(HIDDEN_SIZE, 0);
        vector<double> dbi(HIDDEN_SIZE, 0);
        vector<double> dbc(HIDDEN_SIZE, 0);
        vector<double> dbo(HIDDEN_SIZE, 0);
        
        // Backpropagation through time
        for (int t = sequence_length - 1; t >= 0; t--) {
            // Get activations for current timestep
            vector<double>& o_gate = o_gates[t];
            vector<double>& i_gate = i_gates[t];
            vector<double>& f_gate = f_gates[t];
            vector<double>& c_gate = c_gates[t];
            vector<double>& c = c_states[t + 1];
            vector<double>& prev_c = c_states[t];
            vector<double>& prev_h = h_states[t];
            vector<double>& x = x_inputs[t];
            
            // Concatenate previous hidden state and input
            vector<double> concat;
            concat.insert(concat.end(), prev_h.begin(), prev_h.end());
            concat.insert(concat.end(), x.begin(), x.end());
            
            // Gradients for gates
            vector<double> do_gate(HIDDEN_SIZE);
            vector<double> di_gate(HIDDEN_SIZE);
            vector<double> df_gate(HIDDEN_SIZE);
            vector<double> dc_gate(HIDDEN_SIZE);
            
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                // Output gate gradient
                do_gate[i] = dh[i] * tanh_custom(c[i]) * sigmoid_derivative(o_gate[i]);
                
                // Cell state gradient
                dc[i] += dh[i] * o_gate[i] * tanh_derivative(tanh_custom(c[i]));
                
                // Input gate gradient
                di_gate[i] = dc[i] * c_gate[i] * sigmoid_derivative(i_gate[i]);
                
                // Forget gate gradient
                df_gate[i] = dc[i] * prev_c[i] * sigmoid_derivative(f_gate[i]);
                
                // Cell gate gradient
                dc_gate[i] = dc[i] * i_gate[i] * tanh_derivative(c_gate[i]);
                
                // Accumulate gradients for weights and biases
                for (size_t j = 0; j < concat.size(); j++) {
                    dWf[i][j] += df_gate[i] * concat[j];
                    dWi[i][j] += di_gate[i] * concat[j];
                    dWc[i][j] += dc_gate[i] * concat[j];
                    dWo[i][j] += do_gate[i] * concat[j];
                }
                
                dbf[i] += df_gate[i];
                dbi[i] += di_gate[i];
                dbc[i] += dc_gate[i];
                dbo[i] += do_gate[i];
            }
            
            // Propagate error to previous cell state and hidden state
            vector<double> dc_prev(HIDDEN_SIZE, 0);
            vector<double> dh_prev(HIDDEN_SIZE, 0);
            
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                dc_prev[i] = dc[i] * f_gate[i];
                
                // Gradients to previous hidden state (through all gates)
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    dh_prev[i] += df_gate[j] * Wf[j][i];
                    dh_prev[i] += di_gate[j] * Wi[j][i];
                    dh_prev[i] += dc_gate[j] * Wc[j][i];
                    dh_prev[i] += do_gate[j] * Wo[j][i];
                }
            }
            
            // Update for next iteration
            dh = dh_prev;
            dc = dc_prev;
        }
        
        // Apply gradient clipping
        auto clip_weights = [&](vector<vector<double>>& dW) {
            double norm = 0;
            for (auto& row : dW) {
                for (double val : row) {
                    norm += val * val;
                }
            }
            norm = sqrt(norm);
            
            if (norm > GRADIENT_CLIP) {
                double scale = GRADIENT_CLIP / norm;
                for (auto& row : dW) {
                    for (double& val : row) {
                        val *= scale;
                    }
                }
            }
        };
        
        clip_weights(dWf);
        clip_weights(dWi);
        clip_weights(dWc);
        clip_weights(dWo);
        
        // Update weights and biases
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            for (size_t j = 0; j < INPUT_SIZE + HIDDEN_SIZE; j++) {
                Wf[i][j] -= learning_rate * dWf[i][j];
                Wi[i][j] -= learning_rate * dWi[i][j];
                Wc[i][j] -= learning_rate * dWc[i][j];
                Wo[i][j] -= learning_rate * dWo[i][j];
            }
            
            bf[i] -= learning_rate * dbf[i];
            bi[i] -= learning_rate * dbi[i];
            bc[i] -= learning_rate * dbc[i];
            bo[i] -= learning_rate * dbo[i];
        }
    }
};

// Simple Fully Connected Layer
struct Linear {
    vector<vector<double>> weights;
    vector<double> bias;
    
    // For storing inputs during forward pass
    vector<double> last_input;

    Linear(int input_size, int output_size) {
        weights = vector<vector<double>>(output_size, vector<double>(input_size));
        for (auto& row : weights) for (double& val : row) val = dist(gen);
        bias = vector<double>(output_size, 0.1);
    }

    vector<double> forward(const vector<double>& input) {
        last_input = input;
        vector<double> output(bias);
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < input.size(); j++) {
                output[i] += weights[i][j] * input[j];
            }
        }
        return output;
    }
    
    vector<double> backward(const vector<double>& grad_output, float learning_rate) {
        // Compute gradient for input
        vector<double> grad_input(last_input.size(), 0);
        
        for (size_t i = 0; i < last_input.size(); i++) {
            for (size_t j = 0; j < grad_output.size(); j++) {
                grad_input[i] += grad_output[j] * weights[j][i];
            }
        }
        
        // Update weights and biases
        for (size_t i = 0; i < weights.size(); i++) {
            for (size_t j = 0; j < last_input.size(); j++) {
                weights[i][j] -= learning_rate * grad_output[i] * last_input[j];
            }
            bias[i] -= learning_rate * grad_output[i];
        }
        
        return grad_input;
    }
};

// Load stock price data from CSV
vector<double> loadStockData(const string& filename) {
    vector<double> prices;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return prices;
    }

    string line;
    // Skip header if present
    getline(file, line);
    
    while (getline(file, line)) {
        try {
            if (!line.empty() && isdigit(line[0])) {
                // Split by comma
                vector<string> tokens;
                stringstream ss(line);
                string token;
                while (getline(ss, token, ',')) {
                    tokens.push_back(token);
                }
    
                // Ensure we have at least 5 fields (index 4 = Opening price)
                if (tokens.size() >= 2) {
                    prices.push_back(stod(tokens[1]));  // 5th column = Opening Price
                }
            }
        } catch (const exception& e) {
            cerr << "Error parsing line: " << line << " - " << e.what() << endl;
        }
    }
    
    file.close();

    if (prices.empty()) {
        cerr << "No valid data loaded from file." << endl;
    } else {
        cout << "Loaded " << prices.size() << " data points." << endl;
    }
    
    return prices;
}

// Normalize data using Min-Max normalization (more stable than Z-score for financial data)
void normalize(vector<double>& prices, double& min_val, double& max_val) {
    if (prices.empty()) return;
    
    min_val = *min_element(prices.begin(), prices.end());
    max_val = *max_element(prices.begin(), prices.end());
    
    double range = max_val - min_val;
    if (range < 1e-5) range = 1.0; // Avoid division by near-zero
    
    for (double& price : prices) {
        price = (price - min_val) / range;
    }
}

// Create batches of sequences for training
vector<pair<vector<double>, double>> createSequences(const vector<double>& prices, int seq_length) {
    vector<pair<vector<double>, double>> sequences;
    for (size_t i = 0; i <= prices.size() - seq_length - 1; i++) {
        vector<double> seq(prices.begin() + i, prices.begin() + i + seq_length);
        double target = prices[i + seq_length];
        sequences.push_back({seq, target});
    }
    return sequences;
}

// Train LSTM model
void trainModel(LSTMCell& lstm, Linear& fc, const vector<pair<vector<double>, double>>& sequences) {
    vector<double> loss_history;
    double prev_checkpoint_loss = 1e9;
    int epoch = 0;
    while(1) {
        double total_loss = 0.0;
        int num_samples = 0;

        // Shuffle sequences for each epoch
        vector<pair<vector<double>, double>> shuffled_sequences = sequences;
        shuffle(shuffled_sequences.begin(), shuffled_sequences.end(), gen);

        // Train on each sequence
        for (const auto& seq_pair : shuffled_sequences) {
            const vector<double>& sequence = seq_pair.first;
            double target = seq_pair.second;

            // Forward pass
            lstm.reset_states();
            vector<double> lstm_output = lstm.forward(sequence);
            vector<double> prediction = fc.forward(lstm_output);

            // Compute loss
            double error = prediction[0] - target;
            double loss = error * error;
            total_loss += loss;
            num_samples++;

            // Backward pass
            vector<double> output_grad = {2 * error}; // MSE derivative
            vector<double> hidden_grad = fc.backward(output_grad, LEARNING_RATE);

            // Reset cell state gradient
            vector<double> cell_grad(HIDDEN_SIZE, 0);

            // Update LSTM weights
            lstm.backward(hidden_grad, cell_grad, LEARNING_RATE);
        }

        double avg_loss = total_loss / num_samples;
        loss_history.push_back(avg_loss);

        // Print every 10 epochs
        if (epoch % 10 == 0) {
            cout << "Epoch [" << epoch + 1 << "] "
                 << "Loss: " << fixed << setprecision(6) << avg_loss << endl;

            // Check convergence
            if (epoch >= 10) {
                double loss_diff = abs(prev_checkpoint_loss - avg_loss);
                if (loss_diff < 0.0001) {
                    break;
                }
            }

            prev_checkpoint_loss = avg_loss;
        }
        epoch++;
    }
}


// Make prediction using trained model
double predictNextPrice(LSTMCell& lstm, Linear& fc, const vector<double>& last_sequence, double min_val, double max_val) {
    lstm.reset_states();
    vector<double> lstm_output = lstm.forward(last_sequence);
    vector<double> prediction = fc.forward(lstm_output);
    
    // Denormalize
    return prediction[0] * (max_val - min_val) + min_val;
}

vector<double> evaluateModel(LSTMCell& lstm, Linear& fc, const vector<pair<vector<double>, double>>& test_sequences, const double min_val, const double max_val) {
    vector<double> y_true;
    vector<double> y_pred;

    // Collect true values and predicted values
    for (const auto& seq_pair : test_sequences) {
        const vector<double>& sequence = seq_pair.first;
        double target = seq_pair.second;

        lstm.reset_states();
        vector<double> lstm_output = lstm.forward(sequence);
        vector<double> prediction = fc.forward(lstm_output);

        y_true.push_back(target);
        y_pred.push_back(prediction[0]);
    }

    // R² Score
    double mean_y = accumulate(y_true.begin(), y_true.end(), 0.0) / y_true.size();
    double ss_total = 0.0, ss_res = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        ss_total += (y_true[i] - mean_y) * (y_true[i] - mean_y);
        ss_res += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
    }

    // RMSE Calculation
    double sum_sq = 0.0;
    double scale = max_val - min_val;

    if (scale < 1e-5) scale = 1.0; 

    for (size_t i = 0; i < y_true.size(); ++i) {
        double true_actual = y_true[i] * scale + min_val;
        double pred_actual = y_pred[i] * scale + min_val;

        double diff = true_actual - pred_actual;
        sum_sq += diff * diff;
    }

    // Collect evaluation metrics
    vector<double> eval;
    eval.push_back(1.0 - (ss_res / ss_total)); // R² Score
    eval.push_back(sqrt(sum_sq / y_true.size())); // RMSE

    return eval;
}

int main(int argc, char* argv[]) {
    // Check if filename is provided
    string filename = "ICICIBANK_NS_open_prices_1y.csv";
    if (argc > 1) {
        filename = argv[1];
    }
    
    // Load and preprocess data
    vector<double> stock_prices = loadStockData(filename);
    if (stock_prices.size() < SEQ_LENGTH + 1) {
        cerr << "Not enough data points. Need at least " << SEQ_LENGTH + 1 << " values." << endl;
        return 1;
    }
    
    double min_val, max_val;
    normalize(stock_prices, min_val, max_val);
    
    // Create training sequences
    vector<pair<vector<double>, double>> sequences = createSequences(stock_prices, SEQ_LENGTH);

    // Train-test split (80-20 split)
    size_t train_size = sequences.size() * 0.8;
    vector<pair<vector<double>, double>> train_sequences(sequences.begin(), sequences.begin() + train_size);
    vector<pair<vector<double>, double>> test_sequences(sequences.begin() + train_size, sequences.end());

    // Initialize model
    LSTMCell lstm(INPUT_SIZE, HIDDEN_SIZE);
    Linear fc(HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Train model
    cout << "Training model on " << train_sequences.size() << " sequences..." << endl;
    trainModel(lstm, fc, train_sequences);

    // Evaluate model on test set
    vector<double> eval = evaluateModel(lstm, fc, test_sequences, min_val, max_val);
    cout << "Evaluation on Test Set:" << endl;
    cout << "R2 Score on Test Set: " << fixed << setprecision(6) << eval[0] << endl;    
    cout << "RMSE: " << fixed << setprecision(6) << eval[1] << endl;

    // Predict next price using the last sequence from the training set
    double next_price = predictNextPrice(lstm, fc, sequences.back().first, min_val, max_val);   
    cout << "Predicted next price: " << fixed << setprecision(6) << next_price << endl;
    
    return 0;
}