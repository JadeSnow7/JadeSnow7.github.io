# huaodong.online Deployment (Option 2: Self-hosted)

This repo can be deployed to your Ubuntu server with:
- Nginx serving the static site (same domain)
- FastAPI backend proxied under `/api/*`
- HTTPS via Let's Encrypt (Certbot)

## 1) DNS

Set these records (at your DNS provider):
- `huaodong.online`  -> `A` -> `106.54.188.236`
- `www.huaodong.online` -> `A` -> `106.54.188.236`

## 2) Server packages

```bash
sudo apt-get update
sudo apt-get install -y nginx python3-venv certbot python3-certbot-nginx
```

## 3) Website directory

```bash
sudo mkdir -p /var/www/huaodong.online
sudo chown -R ubuntu:ubuntu /var/www/huaodong.online
```

## 4) Backend directory

```bash
sudo mkdir -p /opt/huaodong-chatbot/backend
sudo chown -R ubuntu:ubuntu /opt/huaodong-chatbot
```

## 5) Nginx config

Copy `deploy/nginx-huaodong.online.conf` to:
- `/etc/nginx/sites-available/huaodong.online`

Then:
```bash
sudo ln -s /etc/nginx/sites-available/huaodong.online /etc/nginx/sites-enabled/huaodong.online
sudo nginx -t
sudo systemctl reload nginx
```

## 6) HTTPS

```bash
sudo certbot --nginx -d huaodong.online -d www.huaodong.online
```

## 7) Backend service (systemd)

Copy `deploy/huaodong-chatbot.service` to:
- `/etc/systemd/system/huaodong-chatbot.service`

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now huaodong-chatbot
sudo systemctl status huaodong-chatbot --no-pager
```

Health check (after backend is running):
```bash
curl -s https://huaodong.online/api/health
```

## 8) Deploying code

Use `deploy_prod.sh` from your local machine:
```bash
./deploy_prod.sh
```

