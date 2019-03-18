import os
import pickle
import time
import torch
import torch.utils
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from .fp16util import network_to_half, set_grad, copy_in_params


def get_best_tmp_path(train_id):
    print("train_id: ", train_id)
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    best_tmp_path = 'tmp/'+str(train_id)+'.model'
    return best_tmp_path


def draw_curve(data, xlabel, ylabel, outputfile):
    plt.figure()
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data)
    plt.savefig(outputfile)


def train_single_input(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                       train_path, epochs, tag_score_func, optimizer=None, return_model=False, train_id=None):
    """
    defaultly return the train_id and the model dump path
    """
    model.cuda()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    if train_id is None:
        train_id = time.time()  # use the time stamp as the training id for checkpointing
    best_tmp_path = get_best_tmp_path(train_id)
    checkpoint_path = best_tmp_path + '.checkpoint'
    checkpoint_epoch = 0
    checkpoint = None
    if os.path.exists(checkpoint_path):
        print('load checkpoint from: ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    elif os.path.exists(best_tmp_path):
        print('load model parameters from: ', best_tmp_path)
        model.load_state_dict(torch.load(best_tmp_path))
    else:
        pass
    loss_function = torch.nn.NLLLoss().cuda()
    epoch_loss_history = []
    training_step_loss_history = []
    training_loss_history = []
    best_epoch_loss = 2 ** 16 - 1
    best_epoch = 0
    final_epoch = 0
    for epoch in range(epochs - checkpoint_epoch):
        accumulated_train_loss = 0
        accumulated_val_loss = 0
        batch_num = 0
        # training
        for step_n, tensors in enumerate(tqdm(train_loader, ncols=10)):
            input_tensor, tag_tensor = tensors
            optimizer.zero_grad()
            batch_x = torch.autograd.Variable(input_tensor.cuda())
            batch_tag = torch.autograd.Variable(tag_tensor.cuda()).squeeze(1)
            tag_scores = tag_score_func(batch_x, model)
            try:
                loss = loss_function(input=tag_scores, target=batch_tag)
            except RuntimeError:
                print(RuntimeError)
                print("batch x: ", batch_x, " | batch tag", batch_tag, " | tag tensor:  ", tag_tensor)
                print(batch_x.size(), batch_tag.size(), tag_tensor.size())
            if step_n < (train_set_len / batch_size) * (1 - val_split):
                step_loss = loss.data.cpu().numpy()
                training_step_loss_history.append(step_loss)
                accumulated_train_loss += step_loss
                loss.backward()
                optimizer.step()
            else:
                # validation
                accumulated_val_loss += loss.data.cpu().numpy()
                batch_num += 1
        train_loss = accumulated_train_loss / batch_num
        val_loss = accumulated_val_loss / batch_num
        print('Epoch: ', epoch, ' | Train Loss: ', train_loss, ' | Val Loss: ', val_loss)
        training_loss_history.append(train_loss)
        epoch_loss_history.append(val_loss)
        # save best
        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_tmp_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
        if epoch - best_epoch > early_stop_patience:
            break
        final_epoch = epoch
    draw_curve(training_step_loss_history, 'Steps', 'Training Error', str(train_id) + '_train_step_curve.png')
    draw_curve(training_loss_history, 'Epochs', 'Training Error', str(train_id) + '_train_curve.png')
    draw_curve(epoch_loss_history, 'Epochs', 'Validation Error', str(train_id) + '_val_curve.png')
    pickle.dump(training_step_loss_history,
                open(str(train_id) + '_train_step_loss_history.pkl', 'wb'))
    pickle.dump(training_loss_history, open(str(train_id) + '_train_loss_history.pkl', 'wb'))
    pickle.dump(epoch_loss_history, open(str(train_id) + '_val_loss_history.pkl', 'wb'))
    if return_model:
        model.load_state_dict(torch.load(best_tmp_path))
        return model
    else:
        return train_id, best_tmp_path, final_epoch + 1


def get_rnn_tag_score(batch_x, model):
    hidden = model.encoder.init_hidden(batch_x.size(0))
    tag_scores = F.log_softmax(model(batch_x, hidden)[0], 1)
    return tag_scores


def get_qrnn_tag_score(batch_x, model):
    hidden = model.encoder.init_hidden(batch_x.size(0))
    tag_scores = F.log_softmax(model(batch_x, hidden)[0][-1], 1)
    return tag_scores


def get_ff_tag_score(batch_x, model):
    return F.log_softmax(model(batch_x), 1)


def train_single_input_classifier_rnn(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                                      train_path, epochs, optimizer=None, return_model=False, **kwargs):
    return train_single_input(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                              train_path, epochs, get_rnn_tag_score, optimizer, return_model, **kwargs)


def train_single_input_classifier_qrnn(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                                       train_path, epochs, optimizer=None, return_model=False, **kwargs):
    return train_single_input(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                              train_path, epochs, get_qrnn_tag_score, optimizer, return_model, **kwargs)


def train_single_input_classifier_ff(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                                     train_path, epochs, optimizer=None, return_model=False, **kwargs):
    return train_single_input(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                              train_path, epochs, get_ff_tag_score, optimizer, return_model, **kwargs)


def train_single_input_mask(model, train_loader, train_set_len, batch_size, val_split, early_stop_patience,
                       train_path, epochs, loss_scale=8, optimizer=None, train_id=None, fp16=False):
    """
    the function to deal with masked input
    besides, using mixed precision learning to speed up
    """
    if fp16:
        model.cuda().half()
        param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]
        for param in param_copy:
            param.requires_grad = True
    else:
        model.cuda()
        param_copy = model.parameters()
    if optimizer is None:
        optimizer = torch.optim.Adam(param_copy)
    else:
        optimizer = optimizer(param_copy)

    if train_id is None:
        train_id = time.time()  # use the time stamp as the training id for checkpointing
    best_tmp_path = get_best_tmp_path(train_id)
    checkpoint_path = best_tmp_path + '.checkpoint'
    checkpoint_epoch = 0
    if os.path.exists(checkpoint_path):
        print('load checkpoint from: ', checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    elif os.path.exists(best_tmp_path):
        print('load model parameters from: ', best_tmp_path)
        model.load_state_dict(torch.load(best_tmp_path))
    else:
        pass
    loss_function = torch.nn.NLLLoss().cuda()
    epoch_loss_history = []
    training_step_loss_history = []
    training_loss_history = []
    best_epoch_loss = 2 ** 16 - 1
    best_epoch = 0
    final_epoch = 0
    for epoch in range(epochs - checkpoint_epoch):
        accumulated_train_loss = 0
        accumulated_val_loss = 0
        batch_num = 0
        # training
        for step_n, tensors in enumerate(tqdm(train_loader, ncols=10)):
            input_tensor, tag_tensor, mask = tensors
            model.zero_grad()
            batch_x = torch.autograd.Variable(input_tensor.cuda())
            batch_mask = torch.autograd.Variable(mask.cuda())
            batch_tag = torch.autograd.Variable(tag_tensor.cuda()).squeeze(1)
            if step_n < (train_set_len / batch_size) * (1 - val_split):
                tag_scores = model(batch_x, batch_mask, training=True)
            else:
                tag_scores = model(batch_x, batch_mask, training=False)
            tag_scores = F.log_softmax(tag_scores, 1)
            try:
                loss = loss_function(input=tag_scores, target=batch_tag)
            except RuntimeError:
                print(RuntimeError)
                print("batch x: ", batch_x, " | batch tag", batch_tag, " | tag tensor:  ", tag_tensor)
                print(batch_x.size(), batch_tag.size(), tag_tensor.size())
            loss = loss * loss_scale
            if step_n < (train_set_len / batch_size) * (1 - val_split):
                numpy_train_loss = loss.data.cpu().numpy()
                training_step_loss_history.append(numpy_train_loss)
                accumulated_train_loss += numpy_train_loss
                loss.backward()
                if fp16:
                    set_grad(param_copy, list(model.parameters()))
                    if loss_scale != 1:
                        for param in param_copy:
                            param.grad.data = param.grad.data / loss_scale
                optimizer.step()
                if fp16:
                    params = list(model.parameters())
                    for i in range(len(params)):
                        params[i].data.copy_(param_copy[i].data)
            else:
                # validation
                accumulated_val_loss += loss.data.cpu().numpy()
                batch_num += 1
        train_loss = accumulated_train_loss / batch_num
        val_loss = accumulated_val_loss / batch_num
        print('Epoch: ', epoch, ' | Train Loss: ', train_loss, ' | Val Loss: ', val_loss)
        training_loss_history.append(train_loss)
        epoch_loss_history.append(val_loss)
        # save best
        if val_loss < best_epoch_loss:
            if isinstance(model, torch.nn.DataParallel):
                best_epoch_loss = val_loss
                best_epoch = epoch
                torch.save(model.module.state_dict(), best_tmp_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
            else:
                best_epoch_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_tmp_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)
        if epoch - best_epoch > early_stop_patience:
            break
        final_epoch = epoch
    draw_curve(training_step_loss_history, 'Steps', 'Training Error', str(train_id) + '_train_step_curve.png')
    draw_curve(training_step_loss_history, 'Epochs', 'Training Error', str(train_id) + '_train_curve.png')
    draw_curve(epoch_loss_history, 'Epochs', 'Validation Error', str(train_id) + '_val_curve.png')
    pickle.dump(training_step_loss_history,
                open(str(train_id) + '_train_step_loss_history.pkl', 'wb'))
    pickle.dump(training_loss_history, open(str(train_id) + '_train_loss_history.pkl', 'wb'))
    pickle.dump(epoch_loss_history, open(str(train_id) + '_val_loss_history.pkl', 'wb'))
    return train_id, best_tmp_path, final_epoch + 1