# Training on AWS EC2

Full training, the latency benchmark and the report ran on one EC2 `g6.xlarge` (NVIDIA L4, 24 GB)
in `us-east-1`. The library has no AWS code: data download, S3 sync and instance management are the
shell steps below. Replace `<bucket>` and `<instance-id>` with your own.

## 1. One-time setup

1. Request a G-instance vCPU quota (`Running On-Demand G and VT instances`, 4 vCPU) first; approval
   can take a day.
2. Create a monthly cost budget (30 USD) with alerts, and a budget action that stops the instance
   (section 5).
3. Create an S3 bucket for artifacts, an IAM user for the AWS CLI on the laptop, an EC2 role with
   S3 access, and an SSH key pair.

## 2. Instance

- AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch, Ubuntu 24.04 (owner `amazon`). Find the
  current ID with:

  ```bash
  aws ec2 describe-images --region us-east-1 --owners amazon \
    --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 24.04*" \
    --query "sort_by(Images,&CreationDate)[-3:].[ImageId,Name]" --output text
  ```

- Type `g6.xlarge`, 100 GiB gp3 root volume used as `DATA_ROOT` (do not use the instance store: it
  is wiped on stop), SSH from your IP only, the S3 role attached.
- Spot works: every job is resumable (section 6).

## 3. Code and data

```bash
sudo mkdir -p /data && sudo chown ubuntu:ubuntu /data
export DATA_ROOT=/data S3_URI=s3://<bucket>/layerwise
cd ~/layerwise && uv sync --extra vision
uv run python -c "import torch; print(torch.cuda.is_available())"

# Aalto dataset, Zenodo 10.5281/zenodo.14996805 (CC-BY-4.0), downloaded straight to the instance
mkdir -p $DATA_ROOT/aalto && cd $DATA_ROOT/aalto
curl -L -o aalto.zip "https://zenodo.org/api/records/14996806/files/Original%20Images%20and%20Manual%20Labels.zip/content"
md5sum aalto.zip    # 9d5ed884aecacb9ada5d0cca7c15d4b7
unzip -q aalto.zip  # must create "Original Images and manual labels/"
cd ~/layerwise && uv run layerwise ingest --profile aws
```

## 4. Run

```bash
uv run layerwise train --profile aws --model yolo
uv run layerwise baseline --profile aws
uv run layerwise train --profile aws --model frcnn
uv run layerwise train --profile aws --model yolo --fold 0 --repro
for k in 0 1 2; do uv run layerwise predict --profile aws --run yolo/outer_$k; done
for m in yolo frcnn contrast; do uv run layerwise benchmark --profile aws --run $m/outer_0; done
uv run layerwise report --profile aws
make sync-up
```

## 5. Cost guardrails

- Set the instance's shutdown behaviour to `stop`, so `sudo shutdown -h now` at the end of a long
  job stops billing:

  ```bash
  aws ec2 modify-instance-attribute --instance-id <instance-id> --instance-initiated-shutdown-behavior stop
  ```

- Chain the end of a run to `make -k sync-up && sudo shutdown -h now`.
- Budget actions at 60% and 80% of the monthly budget stop the instance automatically.

## 6. Interruptions

Each finished job writes `result.json` and is skipped on rerun; interrupted training resumes from
`weights/last.pt`. On a new instance, `make sync-down` restores `runs/`, `predictions/` and
`reports/` from S3, then rerun the same command.

## 7. Clean-up

```bash
make sync-down                                   # on the laptop, with DATA_ROOT set
aws ec2 terminate-instances --instance-ids <instance-id>
aws ec2 describe-volumes  --query "Volumes[].VolumeId" --output text      # expect empty
aws ec2 describe-addresses --query "Addresses[].PublicIp" --output text   # expect empty
aws s3 rb s3://<bucket> --force                  # only when the artifacts are no longer needed
```
