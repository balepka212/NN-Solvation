from torch.utils.data import DataLoader
# from Solvation_1.my_nets.Create_dataset import *
# from Solvation_1.Vectorizers.vectorizers import *
from Solvation_1.my_nets.LinearNet import LinearNet3
from Solvation_1.my_nets.ResNET import ResNet1D, MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
from Solvation_1.my_nets.net_func import *
import pickle as pkl


def Experiment(runs_folder='Example_Lin1',
               net='Lin',
               lr=1e-5,
               solvent_vectorizer='solvent_macro_props1',
               solute_vectorizer='solute_TESA',
               norm_bools=(True, True, True),
               net_dict=None,
               ckp_file=None,
               val_loss_min_input=1e16,
               solvent_test_loss_min_input=1e16,
               solute_test_loss_min_input=1e16,
               start_epoch=0,
               save_epochs=(1, 5, 10, 20, 50, 100, 200, 300, 400, 500) + tuple(range(1000, 100000, 1000)),
               epochs=20

               ):
    """Main function to perform experiment.

     Parameters
    ----------
    runs_folder: str
        Name of folder all the files will be stored
    net: 'Lin' or 'Res'
        Which network to use: LinearNet3 or 1dResNET
    lr: float
        Learning rate value
    solvent_vectorizer: str
        Name of vectorizer for solvent
    solute_vectorizer: str
        Name of vectorizer for solute
    norm_bools: (bool, bool, bool)
        Whether apply normalization to solvent vector, solute vector and G_true vector respectively
    Net_Dict: dict
        Kwargs for Network. Especially ResNET
    epochs: int
        number of epochs to train
    """

    comments = f"""solute: {solvent_vectorizer}
     solute: {solute_vectorizer}
     norm: {norm_bools}
     learning rate: {lr}
     Net_Dict: {net_dict}
     epochs: {epochs}"""

    # Create dataset
    table_v1 = pd.read_table(project_path('Solvation_1/Tables/SS_table_v3.tsv'))
    dataset = SS_Dataset(table_v1, solvent_vectorizer, solute_vectorizer, normalize=norm_bools)
    len_data = dataset.__len__()
    val_data = len_data // 10
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len_data - val_data, val_data])

    solvent_table = pd.read_table(project_path('Solvation_1/Tables/solvent_test_table_v3.tsv'))
    solute_table = pd.read_table(project_path('Solvation_1/Tables/solute_test_table_v3.tsv'))
    solvent_test_dataset = SS_Dataset(solvent_table, solvent_vectorizer, solute_vectorizer,
                                      normalize=norm_bools, show_norm_params=False)
    solute_test_dataset = SS_Dataset(solute_table, solvent_vectorizer, solute_vectorizer,
                                     normalize=norm_bools, show_norm_params=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    solvent_test_loader = DataLoader(solvent_test_dataset, batch_size=46, shuffle=False)
    solute_test_loader = DataLoader(solute_test_dataset, batch_size=64, shuffle=False)

    print(f'train length: {len(train_loader.dataset)}')
    print(f'val length: {len(val_loader.dataset)}')
    print(f'solute test length: {len(solute_test_loader.dataset)}')
    print(f'solvent test length: {len(solvent_test_loader.dataset)}\n')

    for folder in ('Solvation_1/Runs/', 'Solvation_1/Run_results/'):
        try:
            os.makedirs(project_path(folder + runs_folder))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        with open(project_path(folder + runs_folder + '/comments.txt'), 'w') as f:
            f.write(comments)
        with open(project_path(folder + runs_folder + '/norm_params.pkl'), 'wb+') as f:
            pkl.dump(dataset.norm_params, f)

    # Train Network
    x, y = next(iter(dataset))
    print(f'Shape of input: {x.shape}\n')

    if net == 'Res':
        in_feat = next(iter(dataset))[0].shape[0]  # get input tensor size
        model = ResNet1D(in_feat, **net_dict)
    elif net == 'Lin':
        in_feat = next(iter(dataset))[0].shape[-1]  # get size of input vector to create a Linear Network
        model = LinearNet3(in_features=in_feat)

    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print('Training')
    MSE = train(model, train_loader, val_loader, solvent_test_loader, solute_test_loader, loss_function,
                optimizer, epochs=epochs, save_epochs=save_epochs, ckp_path=runs_folder,
                start_epoch=start_epoch, val_loss_min_input=val_loss_min_input, from_start=True)
    print('Finished training!')
    plot_losses(project_path('Solvation_1/Run_results/' + runs_folder + '/run_log.tsv'), save=True)
    return MSE


def Continue_experiment(runs_folder='Example_Lin1',
                        net='Lin',
                        lr=1e-5,
                        solvent_vectorizer='solvent_macro_props1',
                        solute_vectorizer='solute_TESA',
                        norm_bools=(True, True, True),
                        net_dict=None,
                        ckp_file=None,
                        val_loss_min_input=1e16,
                        solvent_test_loss_min_input=1e16,
                        solute_test_loss_min_input=1e16,
                        start_epoch=0,
                        save_epochs=(1, 5, 10, 20, 50, 100, 200, 300, 400, 500) + tuple(range(1000, 100000, 1000)),
                        epochs=20

                        ):
    """Main function to perform experiment.

     Parameters
    ----------
    runs_folder: str
        Name of folder all the files will be stored
    net: 'Lin' or 'Res'
        Which network to use: LinearNet3 or 1dResNET
    lr: float
        Learning rate value
    solvent_vectorizer: str
        Name of vectorizer for solvent
    solute_vectorizer: str
        Name of vectorizer for solute
    norm_bools: (bool, bool, bool)
        Whether apply normalization to solvent vector, solute vector and G_true vector respectively
    Net_Dict: dict
        Kwargs for Network. Especially ResNET
    epochs: int
        number of epochs to train
    """

    comments = f"""
    {'continued' if ckp_file else 'from start'}
     solute: {solvent_vectorizer}
     solute: {solute_vectorizer}
     norm: {norm_bools}
     learning rate: {lr}
     Net_Dict: {net_dict}
     epochs: {epochs}
     """

    # Create dataset
    table_v1 = pd.read_table(project_path('Solvation_1/Tables/SS_table_v3.tsv'))
    dataset = SS_Dataset(table_v1, solvent_vectorizer, solute_vectorizer, normalize=norm_bools)
    len_data = dataset.__len__()
    val_data = len_data // 10
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len_data - val_data, val_data])

    solvent_table = pd.read_table(project_path('Solvation_1/Tables/solvent_test_table_v3.tsv'))
    solute_table = pd.read_table(project_path('Solvation_1/Tables/solute_test_table_v3.tsv'))
    solvent_test_dataset = SS_Dataset(solvent_table, solvent_vectorizer, solute_vectorizer,
                                      normalize=norm_bools, show_norm_params=False)
    solute_test_dataset = SS_Dataset(solute_table, solvent_vectorizer, solute_vectorizer,
                                     normalize=norm_bools, show_norm_params=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    solvent_test_loader = DataLoader(solvent_test_dataset, batch_size=46, shuffle=False)
    solute_test_loader = DataLoader(solute_test_dataset, batch_size=64, shuffle=False)

    print(f'train length: {len(train_loader.dataset)}')
    print(f'val length: {len(val_loader.dataset)}')
    print(f'solute test length: {len(solute_test_loader.dataset)}')
    print(f'solvent test length: {len(solvent_test_loader.dataset)}\n')

    for folder in ('Solvation_1/Runs/', 'Solvation_1/Run_results/'):
        try:
            os.makedirs(project_path(folder + runs_folder))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    try:
        os.makedirs(project_path('Solvation_1/Runs/' + runs_folder + '/run_log_hist'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    checkpoint1 = torch.load(project_path(ckp_file))
    start_epoch = checkpoint1['epoch']
    print(f'Continue from epoch {start_epoch}')
    # write losses to log file
    run_folder = project_path('Solvation_1/Runs/' + runs_folder)
    try:
        with open(run_folder + '/run_log.tsv', 'r') as f:
            previous_run_log = f.readlines()  # run log from previous run
            last_epoch = int(float(previous_run_log[-1].split('\t')[0])) # get last epoch from last run
            if last_epoch + 1 != start_epoch:  # Else just continue appending to file
                for i in range(1, 1000):  # Just random large range
                    # check if copy with number {i} exists
                    file_exists = os.path.exists(run_folder+f'/run_log_hist/run_log{i}.tsv')
                    if not file_exists:  # not exists - make a copy with that name
                        shutil.copyfile(run_folder + '/run_log.tsv', run_folder+f'/run_log_hist/run_log{i}.tsv')
                        break  # break the loop
        with open(run_folder + '/run_log.tsv', 'w') as f2:
            f2.truncate()  # clear the run_log file
        with open(run_folder + '/run_log.tsv', 'a') as f2:
            for line in previous_run_log:  # iterate through previously loaded run_log readlines
                # print(f'l: {line}')
                epoch = line.split('\t')[0]
                if not epoch.isalpha() :
                    if int(float(epoch)) + 1 == start_epoch:  # stop when epoch before start epoch is reached
                        f2.write(line)
                        break
                    else:
                        f2.write(line)
                else:
                    f2.write(line)
    except FileNotFoundError:  # No file - means no previous runs
        # Header row for losses log file
        print('Log file Not Found')


    # Train Network
    x, y = next(iter(dataset))
    print(f'Shape of input: {x.shape}\n')

    if net == 'Res':
        in_feat = next(iter(dataset))[0].shape[0]  # get input tensor size
        model = ResNet1D(in_feat, **net_dict)
    elif net == 'Lin':
        in_feat = next(iter(dataset))[0].shape[-1]  # get size of input vector to create a Linear Network
        model = LinearNet3(in_features=in_feat)

    loss_function = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if ckp_file:
        model, optimizer, start_epoch, losses_min = load_ckp(ckp_file, model, optimizer)
        try:
            val_loss_min_input, solvent_loss_min_input, solute_loss_min_input = losses_min
        except TypeError or ValueError:
            val_loss_min_input = losses_min


    model.train()
    print('Training')
    MSE = train(model, train_loader, val_loader, solvent_test_loader, solute_test_loader, loss_function,
                optimizer, epochs=epochs, ckp_path=runs_folder, start_epoch=start_epoch,
                val_loss_min_input=val_loss_min_input,
                solvent_test_loss_min_input=solvent_loss_min_input,
                solute_test_loss_min_input=solute_loss_min_input,
                from_start=False)
    print('Finished training!')

    plot_losses(project_path('Solvation_1/Run_results/' + runs_folder + '/run_log.tsv'), save=True)
    return MSE
