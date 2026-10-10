# AWS Deployment Guide — EC2 + S3

Matches RFP Section 10 (AWS Deployment Expectations): application hosting
on **EC2**, document storage on **S3**, Docker containerization optional.

## 1. Create the S3 bucket (document/file storage)

```bash
aws s3 mb s3://ai-loan-assistant-docs --region ap-south-1
aws s3api put-bucket-encryption \
  --bucket ai-loan-assistant-docs \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

Upload the synthetic policy docs and applicant files (or let the app do it
via `utils/s3_storage.py` once `S3_BUCKET_NAME` + AWS credentials are set
in `.env`):

```bash
aws s3 cp data/policy_docs/ s3://ai-loan-assistant-docs/policy_docs/ --recursive
aws s3 cp data/applications/ s3://ai-loan-assistant-docs/applications/ --recursive
```

## 2. Launch an EC2 instance (application hosting)

1. AMI: **Ubuntu Server 22.04 LTS**
2. Instance type: **t3.medium** (2 vCPU / 4 GB — enough for Streamlit + FAISS)
3. Security group inbound rules:
   | Type | Port | Source |
   |---|---|---|
   | SSH | 22 | Your IP |
   | Custom TCP | 8501 | 0.0.0.0/0 (Streamlit) |
4. Attach an **IAM instance role** so the app needs no stored credentials. The app now
   *writes* to the bucket (uploaded application PDFs and one object per audit event), so
   read-only access is not enough. Use this least-privilege policy:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       { "Effect": "Allow",
         "Action": ["s3:PutObject", "s3:GetObject"],
         "Resource": "arn:aws:s3:::ai-loan-assistant-docs/*" },
       { "Effect": "Allow",
         "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
         "Resource": "arn:aws:s3:::ai-loan-assistant-docs" }
     ]
   }
   ```

   Create the bucket yourself (Section 1) — the policy deliberately does not allow `s3:CreateBucket`.

## 3. Install dependencies on the instance

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

sudo apt update && sudo apt install -y python3-pip python3-venv git

git clone <your-repo-url> ai-loan-eligibility-assistant
cd ai-loan-eligibility-assistant

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env   # set S3_BUCKET_NAME, AWS_REGION and S3_USE_INSTANCE_ROLE=1 (uses the instance role);
            # OPENAI_API_KEY is optional
```

## 4. Generate data, build the RAG index, and run

```bash
python data_generation/generate_synthetic_data.py
python -m rag.ingest

# Run in background with nohup (simplest option) ...
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

# ...or as a systemd service (recommended, survives reboot):
sudo tee /etc/systemd/system/loan-assistant.service <<'EOF'
[Unit]
Description=AI Loan Eligibility Assessment Assistant
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-loan-eligibility-assistant
ExecStart=/home/ubuntu/ai-loan-eligibility-assistant/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
EnvironmentFile=/home/ubuntu/ai-loan-eligibility-assistant/.env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now loan-assistant
```

Visit `http://<EC2_PUBLIC_IP>:8501` to access the app.

### Verify the S3 integration

```bash
python -m utils.s3_storage          # prints {"configured": true, "reachable": true, ...}
python -m utils.s3_storage sync     # uploads policy PDFs + sample forms
```
In the app: **Settings → AWS S3 Integration** shows the live connection status and has a sync button.
Upload a PDF in **New Assessment**, then check `applications/uploads/<applicant_id>/` and `audit_log/` in the bucket.

## 5. (Optional) Docker-based deployment

```bash
sudo apt install -y docker.io
sudo docker build -t loan-assistant -f deployment/Dockerfile .
sudo docker run -d -p 8501:8501 --env-file .env --name loan-assistant loan-assistant
sudo docker ps   # STATUS should become (healthy) after ~40s
```

## 6. Capture deployment proof for the capstone report

For the mandatory "AWS deployment screenshots" deliverable, capture:
- The EC2 instance details page (instance type, public IP, running state)
- The security group inbound rules
- The S3 bucket contents (policy_docs/ and applications/ prefixes)
- The running Streamlit app in a browser at `http://<EC2_PUBLIC_IP>:8501`
- `systemctl status loan-assistant` (or `docker ps`) showing the service is active
