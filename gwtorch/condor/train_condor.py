import os
import argparse
import shutil
import sys
from pathlib import Path


def resolve_env_executable(command_name):
    env_bin = Path(sys.executable).resolve().parent
    env_command = env_bin / command_name
    if env_command.exists() and os.access(env_command, os.X_OK):
        return str(env_command)

    which_command = shutil.which(command_name)
    if which_command:
        return which_command

    return command_name


def generate_submit_file(
    memory,
    disk,
    cpu,
    model,
    model_subtype,
    classification_scheme,
    epochs,
    batch_size,
    data_file_path,
    accounting_group,
    accounting_group_user,
):
    if classification_scheme == 'binary':
        path=f"submit/{classification_scheme}_{model_subtype}_train.sub"
        executable_name = 'binary_' + model + '_train_gwtorch'
        job_name = f"{classification_scheme}_{model_subtype}_train"
    else:
        path=f"submit/{model_subtype}_train.sub"
        executable_name = model + '_train_gwtorch'
        job_name = f"{model_subtype}_train"
    executable = resolve_env_executable(executable_name)

    """Generates the HTCondor submit file for the model training."""
    content = f"""executable = {executable}

job_name = {job_name}

request_memory = {memory}
request_cpus = {cpu}
request_disk = {disk}

getenv = True

log = HTCondor_Logs/$(job_name).log
output = HTCondor_Logs/$(job_name).output
error = HTCondor_Logs/$(job_name).error

stream_error = True
stream_output = True

accounting_group = {accounting_group}
accounting_group_user = {accounting_group_user}

should_transfer_files = yes
transfer_input_files = results
when_to_transfer_output = ON_EXIT
transfer_output_files = models, results

arguments = --model {model_subtype} --batch-size {batch_size} --epochs {epochs} --path-name {data_file_path}

queue
"""
    with open(path, "w") as f:
        f.write(content)
    return executable

def main():
    parser = argparse.ArgumentParser(
        description="Generate a complete DAGMan workflow with helper scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required argument
    parser.add_argument("--model", type=str, choices=['cnn', 'transformer'],
                        help="Model type for training jobs")
    parser.add_argument("--model-subtype", type=str, choices=['CNN', 'SimpleCNN', 'ViT', 'ViTLinear', 'CvT'],
                        help="Model type for training jobs")
    parser.add_argument("--classification-scheme", type=str, choices=['binary', 'multi-class'],
                        help="Classification scheme for the model")
    
    # Optional arguments for job resources
    parser.add_argument("--memory", type=str, default="20G",
                        help="Memory request for each job (e.g., '8G', '1024M')")
    parser.add_argument("--disk", type=str, default="20G",
                        help="Disk request for each job (e.g., '40G', '2048M')")
    parser.add_argument("--cpu", type=int, default=5,
                        help="Number of CPUs to request for each job")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for training")
    parser.add_argument("--data-file-path", type=str, default="data",
                        help="Path to the dataset directory")
    parser.add_argument("--accounting-group", type=str, default=os.getenv("GWTORCH_ACCOUNTING_GROUP", "iucaa.gw_lens"),
                        help="HTCondor accounting group")
    parser.add_argument("--accounting-group-user", type=str, default=os.getenv("GWTORCH_ACCOUNTING_GROUP_USER", "saqlain.afroz"),
                        help="HTCondor accounting group user")
    
    args = parser.parse_args()
    data_file_path = Path(args.data_file_path).expanduser()
    if not data_file_path.is_absolute():
        data_file_path = Path.cwd() / data_file_path
    data_file_path = data_file_path.resolve()
    
    # Create directories needed for the workflow
    os.makedirs("submit", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("HTCondor_Logs", exist_ok=True)

    if args.model == 'cnn' and args.model_subtype not in ['CNN', 'SimpleCNN']:
        raise ValueError("For CNN model, model-subtype must be 'CNN' or 'SimpleCNN'.")
    if args.model == 'transformer' and args.model_subtype not in ['ViT', 'ViTLinear', 'CvT']:
        raise ValueError("For transformer model, model-subtype must be 'ViT', 'ViTLinear', or 'CvT'.")
    
    # Generate all necessary files
    executable = generate_submit_file(
        memory=args.memory,
        disk=args.disk,
        cpu=args.cpu,
        model=args.model,
        model_subtype=args.model_subtype,
        classification_scheme=args.classification_scheme,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_file_path=str(data_file_path),
        accounting_group=args.accounting_group,
        accounting_group_user=args.accounting_group_user,
    )

    print(f"✅ Successfully generated submit file for training {args.model_subtype} model in the 'submit' directory.")
    print(f"Using executable: {executable}")
    
    if args.classification_scheme == 'binary':
        print(f"To run, execute: condor_submit submit/{args.classification_scheme}_{args.model_subtype}_train.sub")
    else:
        print(f"To run, execute: condor_submit submit/{args.model_subtype}_train.sub")


if __name__ == "__main__":
    main()
