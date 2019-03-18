import os
import pickle
import numpy
import torch
import torch.utils
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from tqdm import tqdm

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from training import get_best_tmp_path
else:
    from .training import get_best_tmp_path


def test_model(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path,
               predict_func):
    best_tmp_path = get_best_tmp_path(train_id)
    model.load_state_dict(torch.load(best_tmp_path))
    model.cuda()
    test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments = \
        predict_func(test_set_len, test_batch, tag_size, test_loader, model)
    pickle.dump(test_predictions, open(os.path.join(test_path, str(train_id) + '_test_predictions.pkl'), 'wb'))
    pickle.dump(test_judgments, open(os.path.join(test_path, str(train_id) + '_test_judgments.pkl'), 'wb'))
    pickle.dump(test_labels, open(os.path.join(test_path, str(train_id) + '_test_labels.pkl'), 'wb'))
    test_loss = accumulated_loss / batch_num
    print('Testing Error: ', test_loss)
    print('Accuracy: ', accuracy_score(true_labels, test_labels))
    print(classification_report(true_labels, test_labels, digits=6))
    return true_labels, test_labels


def predict_single_tesnor(test_set_len, test_batch, tag_size, test_loader, model, predict_func):
    loss_function = torch.nn.NLLLoss().cuda()
    test_predictions = numpy.zeros((test_set_len, tag_size))
    test_judgments = numpy.zeros(test_set_len)
    test_labels = numpy.zeros(test_set_len)
    true_labels = numpy.zeros(test_set_len)
    accumulated_loss = 0
    batch_num = 0
    correct = 0
    total = 0
    print('Testing ')
    for step_n, tensors in enumerate(tqdm(test_loader, ncols=10)):
        input_tensor, tag_tensor = tensors
        batch_x = torch.autograd.Variable(input_tensor.cuda())
        batch_tag = torch.autograd.Variable(tag_tensor.cuda()).squeeze(1)
        tag_prediction = predict_func(batch_x, model)
        tag_scores = F.log_softmax(tag_prediction, 1)
        loss = loss_function(tag_scores, batch_tag)
        accumulated_loss += loss.data.cpu().numpy()
        batch_num += 1
        _, predicted = torch.max(tag_scores.data, 1)
        total += tag_tensor.size(0)
        correct += (predicted == batch_tag).sum().item()
        test_predictions[step_n * test_batch: (step_n + 1) * test_batch] = tag_prediction.data.cpu().numpy()
        test_judgments[step_n * test_batch: (step_n + 1) * test_batch] = (predicted == batch_tag).data.cpu().numpy()
        test_labels[step_n * test_batch: (step_n + 1) * test_batch] = predicted.data.cpu().numpy()
        true_labels[step_n * test_batch: (step_n + 1) * test_batch] = tag_tensor.squeeze(1).numpy()
    return test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments


def get_rnn_predict(x, model):
    hidden = model.encoder.init_hidden(x.size(0))
    tag_prediction = model(x, hidden)[0]
    return tag_prediction


def get_ff_predict(x, model):
    tag_prediction = model(x)
    return tag_prediction


def get_qrnn_predict(batch_x, model):
    hidden = model.encoder.init_hidden(batch_x.size(0))
    prediction = model(batch_x, hidden)[0][-1]
    return prediction


def predict_rnn(test_set_len, test_batch, tag_size, test_loader, model):
    test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments \
        = predict_single_tesnor(test_set_len, test_batch, tag_size, test_loader, model, get_rnn_predict)
    return test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments


def predict_qrnn(test_set_len, test_batch, tag_size, test_loader, model):
    test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments \
        = predict_single_tesnor(test_set_len, test_batch, tag_size, test_loader, model, get_qrnn_predict)
    return test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments


def predict_ff(test_set_len, test_batch, tag_size, test_loader, model):
    test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments \
        = predict_single_tesnor(test_set_len, test_batch, tag_size, test_loader, model, get_ff_predict)
    return test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments


def predict_rnn_att(test_set_len, test_batch, tag_size, test_loader, model):
    loss_function = torch.nn.NLLLoss().cuda()
    test_predictions = numpy.zeros((test_set_len, tag_size))
    test_judgments = numpy.zeros(test_set_len)
    test_labels = numpy.zeros(test_set_len)
    true_labels = numpy.zeros(test_set_len)
    accumulated_loss = 0
    batch_num = 0
    correct = 0
    total = 0
    print('Testing ')
    for step_n, tensors in enumerate(test_loader):
        input_tensor, tag_tensor = tensors
        batch_x = torch.autograd.Variable(input_tensor.cuda())
        batch_tag = torch.autograd.Variable(tag_tensor.cuda()).squeeze(1)
        tag_predictions, hidden, att_weights = model(batch_x)
        tag_scores = F.log_softmax(tag_predictions, 1)
        loss = loss_function(tag_scores, batch_tag)
        accumulated_loss += loss.data.cpu().numpy()
        batch_num += 1
        _, predicted = torch.max(tag_scores.data, 1)
        total += tag_tensor.size(0)
        correct += (predicted == batch_tag).sum().item()
        test_predictions[step_n * test_batch: (step_n + 1) * test_batch] = tag_predictions.data.cpu().numpy()
        test_judgments[step_n * test_batch: (step_n + 1) * test_batch] = (predicted == batch_tag).data.cpu().numpy()
        test_labels[step_n * test_batch: (step_n + 1) * test_batch] = predicted.data.cpu().numpy()
        true_labels[step_n * test_batch: (step_n + 1) * test_batch] = tag_tensor.squeeze(1).numpy()
    return test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments


def test_single_input_classifier_rnn(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path):
    return test_model(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path,
                      predict_rnn)


def test_single_input_classifier_qrnn(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path):
    return test_model(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path,
                      predict_qrnn)


def test_single_input_classifier_ff(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path):
    return test_model(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path,
                      predict_ff)


def test_single_input_classifier_rnn_att(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path):
    return test_model(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path,
                      predict_rnn_att)


def predict_single_tesnor_mask(test_set_len, test_batch, tag_size, test_loader, model):
    loss_function = torch.nn.NLLLoss().cuda()
    test_predictions = numpy.zeros((test_set_len, tag_size))
    test_judgments = numpy.zeros(test_set_len)
    test_labels = numpy.zeros(test_set_len)
    true_labels = numpy.zeros(test_set_len)
    accumulated_loss = 0
    batch_num = 0
    correct = 0
    total = 0
    print('Testing ')
    for step_n, tensors in enumerate(tqdm(test_loader, ncols=10)):
        input_tensor, tag_tensor, mask = tensors
        batch_x = torch.autograd.Variable(input_tensor.cuda())
        batch_mask = torch.autograd.Variable(mask.cuda())
        batch_tag = torch.autograd.Variable(tag_tensor.cuda()).squeeze(1)
        tag_prediction = model(batch_x, batch_mask, training=False)
        tag_scores = F.log_softmax(tag_prediction, 1)
        loss = loss_function(tag_scores, batch_tag)
        accumulated_loss += loss.data.cpu().numpy()
        batch_num += 1
        _, predicted = torch.max(tag_scores.data, 1)
        total += tag_tensor.size(0)
        correct += (predicted == batch_tag).sum().item()
        test_predictions[step_n * test_batch: (step_n + 1) * test_batch] = tag_prediction.data.cpu().numpy()
        test_judgments[step_n * test_batch: (step_n + 1) * test_batch] = (predicted == batch_tag).data.cpu().numpy()
        test_labels[step_n * test_batch: (step_n + 1) * test_batch] = predicted.data.cpu().numpy()
        true_labels[step_n * test_batch: (step_n + 1) * test_batch] = tag_tensor.squeeze(1).numpy()
    return test_predictions, test_labels, accumulated_loss, batch_num, correct, total, true_labels, test_judgments


def test_single_input_classifier_mask(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path):
    return test_model(train_id, model, test_loader, test_batch, test_set_len, tag_size, test_path,
                      predict_single_tesnor_mask)