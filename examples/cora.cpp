#include "autogradpp/mlp.hpp"
#include "autogradpp/operands.hpp"
#include <autogradpp/autogradpp.hpp>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <unordered_map>

using namespace autogradpp;

struct Paper {
    std::size_t id;
    Tensor attributes;
    size_t label;
};

class Dataset {
public:
    Dataset(const std::string& dir) {
        load_content(dir + "/cora.content");
        load_cites(dir + "/cora.cites");
    }

    size_t num_papers() const { return _papers.size(); }
    size_t num_classes() const { return _classes.size(); }
    size_t num_attributes() const { return _num_attributes; }
    Paper get_paper(size_t id) const { return _papers.at(id); }
    std::string get_class(size_t id) const { return _classes.at(id); }
    std::vector<size_t> neighbors(size_t id) { return _neighbors.at(id); }

    Paper operator[](size_t idx) { 
        size_t id = _paper_order[idx];
        return get_paper(id); 
    }
private:
    std::unordered_map<size_t, Paper> _papers;
    std::vector<size_t> _paper_order;
    std::vector<std::string> _classes;
    std::unordered_map<size_t, std::vector<size_t>> _neighbors;
    std::vector<std::pair<size_t, size_t>> _adj_list;
    size_t _num_attributes;

    void load_cites(const std::string& path) {
        std::ifstream fp(path);
        if (!fp) { throw std::runtime_error("Failed to open file for reading: " + path); }
    
        size_t a, b;
        while (fp >> a >> b) {
            auto& a_vec = _neighbors[a];
            auto& b_vec = _neighbors[b];
            a_vec.push_back(b);
            b_vec.push_back(a);
        }
    }

    void load_content(const std::string& path) {
        std::ifstream fp(path);
        if (!fp) { throw std::runtime_error("Failed to open file for reading: " + path); }

        _num_attributes = 0;

        std::string line;
        while (std::getline(fp, line)) {
            std::istringstream iss(line);
            std::vector<std::string> fields; 
            std::string field;

            while (iss >> field) {
                fields.push_back(field);
            }

            if (fields.size() < 2) { continue; } // Ignore this line

            // First entry: id, last entry: label
            size_t id = std::stoi(fields[0]);
            std::string label = fields[fields.size()-1];

            size_t label_id = 0;
            for (; label_id < _classes.size(); ++label_id) {
                if (_classes[label_id] == label) { break; }
            }
            if (label_id >= _classes.size()) { _classes.push_back(label); }

            // Get attributes (rest of entries)
            Tensor attributes = Tensor::zeros({fields.size() - 2});
            for (size_t i = 1; i < fields.size() - 1; ++i) {
                attributes[i-1] = std::stod(fields[i]);
            }

            if (_num_attributes > 0 && _num_attributes != attributes.size()) {
                throw std::runtime_error("Missing content!");
            }
            _num_attributes = attributes.size();

            _papers[id] = { id, attributes, label_id };
            _paper_order.push_back(id);
        }

        fp.close();
    }
};

class GNN {
public:
    GNN(size_t num_layers, size_t num_attributes, size_t hidden_dim, size_t num_classes) 
        : _num_layers(num_layers), _num_attributes(num_attributes), _hidden_dim(hidden_dim), _num_classes(num_classes) {
        Tensor w = Tensor::randn({_hidden_dim, _num_attributes}, 0.0, std::sqrt(2.0 / _num_attributes));
        _weights.push_back(Variable(w));

        for (size_t i = 1; i < num_layers; ++i) {
            Tensor w = Tensor::randn({_hidden_dim, _hidden_dim}, 0.0, std::sqrt(2.0 / _hidden_dim));
            _weights.push_back(Variable(w));
        }

        Tensor wc = Tensor::randn({_num_classes, _hidden_dim}, 0.0, std::sqrt(1.0 / _hidden_dim));
        _weights.push_back(Variable(wc));
    }

    std::vector<Variable>& parameters() { return _weights; }
    std::unordered_map<size_t, Variable> forward(Dataset& dataset, const std::vector<size_t>& papers) {
        // Construct the graph
        std::unordered_map<size_t, Variable> embeddings;
        for (size_t i = 0; i < papers.size(); ++i) {
            size_t id = papers[i];
            Paper paper = dataset.get_paper(id);
            embeddings.try_emplace(id, Variable(paper.attributes));
            auto nbhrs = dataset.neighbors(id);
            for (auto& nbhr : nbhrs) {
                embeddings.try_emplace(nbhr, Variable(dataset.get_paper(nbhr).attributes));
            }
        }

        // Perform each message passing layer
        for (size_t layer = 0; layer < _num_layers; ++layer) {
            std::unordered_map<size_t, Variable> new_embeddings;
            for (size_t i = 0; i < papers.size(); ++i) {
                size_t id = papers[i];

                auto nbhrs = dataset.neighbors(id);
                Variable msg = embeddings[id];
                for (auto& nbhr : nbhrs) {
                    if (embeddings.count(nbhr)) { msg += embeddings[nbhr]; }
                }
                msg /= (nbhrs.size() + 1);
                new_embeddings[id] = relu(matmul(_weights[layer], msg));
                if (layer > 0) { new_embeddings[id] += embeddings[id]; } // Residual connection
            }
            embeddings = new_embeddings;
        }

        // Perform the classification
        for (size_t i = 0; i < papers.size(); ++i) {
            size_t id = papers[i];
            auto logits = matmul(_weights[_num_layers], embeddings[id]);
            embeddings[id] = logits;
        }
        
        return embeddings;
    }
private:
    std::vector<Variable> _weights;
    size_t _num_attributes;
    size_t _hidden_dim;
    size_t _num_layers;
    size_t _num_classes;
};

int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    // Create a random generation
    std::mt19937 rng(std::random_device{}());

    // Load the dataset and split 70% training, 30% testing
    Dataset dataset("cora");
    double training_size = ((double)dataset.num_papers()) * 0.7;
    std::vector<size_t> all_papers;
    for (size_t i = 0; i < dataset.num_papers(); ++i) {
        all_papers.push_back(dataset[i].id);
    }
    std::vector<size_t> training_set(all_papers.begin(), all_papers.begin() + training_size);
    std::vector<size_t> testing_set(all_papers.begin() + training_size, all_papers.end());
    
    // Model parameters
    double learning_rate = 1.0;
    size_t num_epochs = 350;

    std::cout << "Training Set: " << training_set.size() << " papers" << std::endl;
    double avg_nbhrs = 0;
    for (auto& id : training_set) { avg_nbhrs += dataset.neighbors(id).size(); }
    std::cout << "Average Neighbors: " << avg_nbhrs / training_set.size() << std::endl;
    std::cout << "Number of Attributes: " << dataset.num_attributes() << std::endl;
    std::cout << "Number of Classes: " << dataset.num_classes() << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Number of Epochs: " << num_epochs << std::endl;
    std::cout << std::endl;

    // Build a GNN with two message passing layers
    GNN gnn(2, dataset.num_attributes(), 64, dataset.num_classes());

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Create GNN
        auto total_loss = Variable(0.0);
        auto y_preds = gnn.forward(dataset, training_set);
        for (auto& id : training_set) {
            Tensor label_onehot = Tensor::zeros({dataset.num_classes()});
            label_onehot[dataset.get_paper(id).label] = 1;
            auto y_true = Constant(label_onehot);

            // Loss function
            auto loss = cross_entropy(softmax(y_preds[id]), y_true);
            total_loss += loss;
        }
        total_loss /= training_set.size();

        // Perform gradient descent
        total_loss.backward();
        for (auto& w : gnn.parameters()) {
            w.value() -= learning_rate * w.grad(); 
            w.grad() = Tensor::zeros(w.grad().shape());
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == num_epochs-1) {
            std::cout << "  epoch " << epoch+1 << " loss=" << total_loss.value() << std::endl;
        }
    }

    // Test on testing set
    std::cout << std::endl;
    std::cout << "== Testing ==" << std::endl;
    double score = 0.0;
    auto y_preds = gnn.forward(dataset, testing_set);
    #pragma omp parallel for
    for (size_t id = 0; id < testing_set.size(); ++id) {
        Paper paper = dataset.get_paper(testing_set[id]);
        auto y_pred = y_preds[paper.id];
        
        int guess = 0;
        double guess_confidence = -10000;
        for (size_t i = 0; i < dataset.num_classes(); ++i) {
            if (y_pred.value()[i] > guess_confidence) {
                guess = i;
                guess_confidence = y_pred.value()[i];
            }
        }

        if (guess == paper.label) { 
            #pragma omp atomic
            score += 1.0; 
        }
    }

    std::cout << std::endl;
    std::cout << "Accuracy: " << (score / double(testing_set.size())) * 100.0 << "%" << std::endl;

    return 0;
}
