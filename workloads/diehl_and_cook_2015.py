import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t
from sklearn.metrics import confusion_matrix

from bindsnet.network import load
from bindsnet.learning import NoOp
from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import load
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.evaluation import assign_labels, update_ngram_scores
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance

from experiments import ROOT_DIR
from experiments.utils import update_curves, print_results

model = 'diehl_and_cook_2015'
data = 'mnist'

data_path = os.path.join(ROOT_DIR, 'data', 'MNIST')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
spikes_path = os.path.join(ROOT_DIR, 'spikes', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, spikes_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, n_neurons=100, n_train=60000, n_test=10000, inhib=100, lr=1e-2, lr_decay=1, time=350, dt=1,
         theta_plus=0.05, tc_theta_decay=1e7, intensity=1, progress_interval=10, update_interval=250, plot=False,
         train=True, gpu=False):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                            'No. examples must be divisible by update_interval'

    params = [
        seed, n_neurons, n_train, inhib, lr, lr_decay, time, dt, theta_plus,
        tc_theta_decay, intensity, progress_interval, update_interval
    ]

    test_params = [
        seed, n_neurons, n_train, n_test, inhib, lr, lr_decay, time, dt,
        theta_plus, tc_theta_decay, intensity, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    n_examples = n_train if train else n_test
    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
    n_classes = 10

    # Build network.
    if train:
        network = DiehlAndCook2015v2(
            n_inpt=784, n_neurons=n_neurons, inh=inhib, dt=dt, norm=78.4,
            theta_plus=theta_plus, tc_theta_decay=tc_theta_decay, nu=[0, lr]
        )

    else:
        network = load(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].tc_theta_decay = 0
        network.layers['Y'].theta_plus = 0

    # Load MNIST data.
    dataset = MNIST(path=data_path, download=True)

    if train:
        images, labels = dataset.get_train()
    else:
        images, labels = dataset.get_test()

    images = images.view(-1, 784)
    images *= intensity

    # Record spikes during the simulation.
    spike_record = torch.zeros(update_interval, time, n_neurons)
    full_spike_record = torch.zeros(n_examples, n_neurons).long()

    # Neuron assignments and spike proportions.
    if train:
        assignments = -torch.ones_like(torch.Tensor(n_neurons))
        proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        ngram_scores = {}
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }

    if train:
        best_accuracy = 0

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)

    # Train the network.
    if train:
        print('\nBegin training.\n')
    else:
        print('\nBegin test.\n')

    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    assigns_im = None
    perf_ax = None

    start = t()
    for i in range(n_examples):
        if i % progress_interval == 0:
            print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
            start = t()

        if i % update_interval == 0 and i > 0:
            if train:
                network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

            if i % len(labels) == 0:
                current_labels = labels[-update_interval:]
            else:
                current_labels = labels[i % len(images) - update_interval:i % len(images)]

            # Update and print accuracy evaluations.
            curves, preds = update_curves(
                curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
                proportions=proportions, ngram_scores=ngram_scores, n=2
            )
            print_results(curves)

            for scheme in preds:
                predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

            # Save accuracy curves to disk.
            to_write = ['train'] + params if train else ['test'] + params
            f = '_'.join([str(x) for x in to_write]) + '.pt'
            torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

            if train:
                if any([x[-1] > best_accuracy for x in curves.values()]):
                    print('New best accuracy! Saving network parameters to disk.')

                    # Save network to disk.
                    network.save(os.path.join(params_path, model_name + '.pt'))
                    path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
                    torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))
                    best_accuracy = max([x[-1] for x in curves.values()])

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(spike_record, current_labels, n_classes, rates)

                # Compute ngram scores.
                ngram_scores = update_ngram_scores(spike_record, current_labels, n_classes, 2, ngram_scores)

            print()

        # Get next input sample.
        image = images[i % len(images)]
        sample = poisson(datum=image, time=time, dt=dt)
        inpts = {'X': sample}

        # Run the network on the input.
        network.run(inpts=inpts, time=time)

        retries = 0
        while spikes['Y'].get('s').sum() < 1 and retries < 3:
            retries += 1
            image *= 2
            sample = poisson(datum=image, time=time, dt=dt)
            inpts = {'X': sample}
            network.run(inpts=inpts, time=time)

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes['Y'].get('s').t()
        full_spike_record[i] = spikes['Y'].get('s').t().sum(0).long()

        # Optionally plot various simulation information.
        if plot:
            _input = image.view(28, 28)
            reconstruction = inpts['X'].view(time, 784).sum(0).view(28, 28)
            _spikes = {layer: spikes[layer].get('s') for layer in spikes}
            input_exc_weights = network.connections[('X', 'Y')].w
            square_weights = get_square_weights(input_exc_weights.view(784, n_neurons), n_sqrt, 28)
            square_assignments = get_square_assignments(assignments, n_sqrt)

            inpt_axes, inpt_ims = plot_input(_input, reconstruction, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(curves, ax=perf_ax)

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    i += 1

    if i % len(labels) == 0:
        current_labels = labels[-update_interval:]
    else:
        current_labels = labels[i % len(images) - update_interval:i % len(images)]

    # Update and print accuracy evaluations.
    curves, preds = update_curves(
        curves, current_labels, n_classes, spike_record=spike_record, assignments=assignments,
        proportions=proportions, ngram_scores=ngram_scores, n=2
    )
    print_results(curves)

    for scheme in preds:
        predictions[scheme] = torch.cat((predictions[scheme], preds[scheme]), -1)

    if train:
        if any([x[-1] > best_accuracy for x in curves.values()]):
            print('New best accuracy! Saving network parameters to disk.')

            # Save network to disk.
            if train:
                network.save(os.path.join(params_path, model_name + '.pt'))
                path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
                torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')

    print('Average accuracies:\n')
    for scheme in curves.keys():
        print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

    # Save accuracy curves to disk.
    to_write = ['train'] + params if train else ['test'] + params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

    # Save results to disk.
    results = [
        np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
        np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
    ]

    to_write = params + results if train else test_params + results
    to_write = [str(x) for x in to_write]
    name = 'train.csv' if train else 'test.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(results_path, name), 'w') as f:
            if train:
                f.write(
                    'random_seed,n_neurons,n_train,inhib,lr,lr_decay,time,timestep,theta_plus,tc_theta_decay,intensity,'
                    'progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                    'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
                )
            else:
                f.write(
                    'random_seed,n_neurons,n_train,n_test,inhib,lr,lr_decay,time,timestep,theta_plus,tc_theta_decay,'
                    'intensity,progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                    'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
                )

    with open(os.path.join(results_path, name), 'a') as f:
        f.write(','.join(to_write) + '\n')

    if labels.numel() > n_examples:
        labels = labels[:n_examples]
    else:
        while labels.numel() < n_examples:
            if 2 * labels.numel() > n_examples:
                labels = torch.cat([labels, labels[:n_examples - labels.numel()]])
            else:
                labels = torch.cat([labels, labels])

    # Compute confusion matrices and save them to disk.
    confusions = {}
    for scheme in predictions:
        confusions[scheme] = confusion_matrix(labels, predictions[scheme])

    to_write = ['train'] + params if train else ['test'] + test_params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save(confusions, os.path.join(confusion_path, f))

    # Save full spike record to disk.
    torch.save(full_spike_record, os.path.join(spikes_path, f))


if __name__ == '__main__':
    print()

    # Parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n_neurons', type=int, default=100, help='no. of output layer neurons')
    parser.add_argument('--n_train', type=int, default=60000, help='no. of training samples')
    parser.add_argument('--n_test', type=int, default=10000, help='no. of test samples')
    parser.add_argument('--inhib', type=float, default=100.0, help='inhibition connection strength')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='rate at which to decay learning rate')
    parser.add_argument('--time', default=350, type=int, help='simulation time')
    parser.add_argument('--dt', type=float, default=1, help='simulation integreation timestep')
    parser.add_argument('--theta_plus', type=float, default=0.05, help='adaptive threshold increase post-spike')
    parser.add_argument('--tc_theta_decay', type=float, default=1e7, help='adaptive threshold decay time constant')
    parser.add_argument('--intensity', type=float, default=0.5, help='constant to multiple input data by')
    parser.add_argument('--progress_interval', type=int, default=10, help='interval to print train, test progress')
    parser.add_argument('--update_interval', default=250, type=int, help='no. examples between evaluation')
    parser.add_argument('--plot', dest='plot', action='store_true', help='visualize spikes + connection weights')
    parser.add_argument('--train', dest='train', action='store_true', help='train phase')
    parser.add_argument('--test', dest='train', action='store_false', help='test phase')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='whether to use cpu or gpu tensors')
    parser.set_defaults(plot=False, gpu=False, train=True)
    args = parser.parse_args()

    args = vars(args)

    print()
    print('Command-line argument values:')
    for key, value in args.items():
        print('-', key, ':', value)

    print()

    main(**args)

    print()
