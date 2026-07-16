# Upload PSX Bot to GitHub and Deploy on Streamlit Cloud

## Step 1 — Extract this ZIP

Download this ZIP and extract it on your computer.

Do not upload this ZIP file directly to GitHub.
Upload the extracted files and folders.

## Step 2 — Create GitHub Repository

1. Go to GitHub.
2. Click the **+** button.
3. Click **New repository**.
4. Repository name: `psx-bot`
5. Choose **Private** if you do not want others to see your bot.
6. Click **Create repository**.

## Step 3 — Upload Files to GitHub

1. Open your new repository.
2. Click **Add file**.
3. Click **Upload files**.
4. Select all extracted files and folders from this project.
5. Click **Commit changes**.

Important: `app.py` must be visible in the main/root folder of the repository.

## Step 4 — Deploy on Streamlit Cloud

Use these settings:

- Repository: `psx-bot`
- Branch: `main`
- Main file path: `app.py`

Then click **Deploy**.

## If Streamlit Shows Module Missing Error

1. Open `requirements.txt`.
2. Add the missing package name.
3. Commit the change.
4. Reboot the Streamlit app.

## Android App Use

After deployment, copy your Streamlit Cloud URL and paste it into the Android APK wrapper app.
