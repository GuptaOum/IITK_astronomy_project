# Launch a g4dn.xlarge GPU instance, upload the dataset, train, download the
# model, and TERMINATE the instance. Run AFTER the GPU quota is approved.
#
# Usage (from the windowpynbody2 folder):
#   powershell -ExecutionPolicy Bypass -File code\aws_train.ps1
#
# Requires: aws cli configured, ~/.ssh/face-attendance.pem, dataset/ rendered.

$ErrorActionPreference = "Stop"

$AMI = "ami-0db16d2662b105f73"   # Deep Learning AMI GPU TensorFlow 2.18 (Ubuntu 22.04)
$TYPE = "g4dn.xlarge"
$KEY = "face-attendance"
$PEM = "$env:USERPROFILE\.ssh\face-attendance.pem"
$SG = "sg-0dca629c63e847a32"     # face-attendance-sg (SSH from your IP)
$NAME = "iitk-bar-cnn-training"

# --- 1. Launch ---
Write-Host "Launching $TYPE ..."
$ID = aws ec2 run-instances `
    --image-id $AMI --instance-type $TYPE `
    --key-name $KEY --security-group-ids $SG `
    --block-device-mappings '[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":60,\"VolumeType\":\"gp3\"}}]' `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" `
    --query "Instances[0].InstanceId" --output text
Write-Host "Instance: $ID"

aws ec2 wait instance-running --instance-ids $ID
$IP = aws ec2 describe-instances --instance-ids $ID `
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text
Write-Host "Public IP: $IP  (waiting 90s for SSH to come up)"
Start-Sleep -Seconds 90

$SSH = "ssh -i `"$PEM`" -o StrictHostKeyChecking=accept-new ubuntu@$IP"

# --- 2. Upload dataset + training script ---
Write-Host "Uploading dataset and training script ..."
tar -czf dataset.tar.gz dataset
scp -i "$PEM" -o StrictHostKeyChecking=accept-new dataset.tar.gz code/train_resnet.py ubuntu@${IP}:~
Remove-Item dataset.tar.gz

# --- 3. Train on the GPU ---
Write-Host "Training (output streams below) ..."
Invoke-Expression "$SSH 'tar xzf dataset.tar.gz && mkdir -p code && mv train_resnet.py code/ && (source activate tensorflow 2>/dev/null || source /opt/tensorflow/bin/activate) && python code/train_resnet.py'"

# --- 4. Download results ---
Write-Host "Downloading models/ ..."
scp -i "$PEM" -r ubuntu@${IP}:~/models .

# --- 5. TERMINATE (billing stops) ---
Write-Host "Terminating $ID ..."
aws ec2 terminate-instances --instance-ids $ID | Out-Null
aws ec2 wait instance-terminated --instance-ids $ID
Write-Host "Done. Model + report are in .\models\"
