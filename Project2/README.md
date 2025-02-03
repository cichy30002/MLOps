## Project 2: Hyper-parameter optimization

The goal of this homework will be to familiarize yourself with Pytorch Lightning which serves as a tool for organization of all-things related to training (this is a semi-quote) and perform hyper-parameter optimization.

If it comes to Pytorch Lightning this will be one time, where it is required to use this particular package without option to chose any other, as I feel there are no other alternatives, (at least for Pytorch). Of course I'm open to proving me wrong.

### Example flow to complete the homework:

* Prepare a training script of any chosen model in Pytorch Lightning. To achieve that you need to create two modules: Data Module and Lightning Module
* Add model monitoring using built-in Pytorch Lighting callback for a chosen monitoring tool, for instance Wandb or MLFlow.
* Validate Model Learning: Confirm the model does in-fact learn by comparison of training to validation loss curve.
* Implement optimization of any chosen (or group of parameters) using a chosen optimization package. (I recommend Optuna or RayTune)


### Bonus point
1 Bonus points will be applied to groups who will dockerize the Pytorch Lightning Trainer, and enable hyper-parameter optimization in parallel with customized logging to a chosen model monitoring Service.

Please note: Optuna offers a callback that automatically logs your run history to a chosen model monitoring tool, but the goal of this task is to enable custom model monitoring in parallel, which requires for the training runs to be in separate processes.  (Access to multiple GPUs is not required)

### Additional organization information:
* If possible, please don’t copy outright whole solutions, like the one linked below.
* Please upload your code to a repository.
* Feel free to use models from any of your previous or future (like Master's degree) projects.
* Deadline: Two week from now



### Useful links:
* https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning
* https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html
* https://freedium.cfd/https://pub.aimind.so/optimizing-hyperparameters-with-optuna-a-hands-on-tutorial-bb8e2ffd1801
* https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
* https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html