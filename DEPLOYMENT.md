# Deployment Guide: GitHub Pages

This guide walks you through deploying your DOS-style portfolio to GitHub Pages.

## Prerequisites
1.  **Git**: Ensure Git is installed (`git --version`).
2.  **GitHub Account**: You need an account on [github.com](https://github.com).
3.  **Node.js**: You already have this installed.

## Step 1: Initialize Git
Since your project is not yet a Git repository, initialize it:

```bash
# Initialize git
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit"
```

## Step 2: Create a GitHub Repository
1.  Go to [GitHub.com](https://github.com) and sign in.
2.  Click the **+** icon in the top-right and select **New repository**.
3.  Name it (e.g., `portfolio-dos`).
4.  Make it **Public**.
5.  Click **Create repository**.

## Step 3: Link and Push
Copy the commands provided by GitHub under "…or push an existing repository from the command line":

```bash
# Replace <YOUR_USERNAME> with your actual GitHub username
git remote add origin https://github.com/<YOUR_USERNAME>/portfolio-dos.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy
We have already configured the `deploy` script in `package.json`. Run:

```bash
npm run deploy
```

This command will:
1.  Build the project (`npm run build`).
2.  Push the `dist` folder to a `gh-pages` branch on your repository.

## Step 5: Verify
1.  Go to your repository on GitHub.
2.  Click **Settings** > **Pages**.
3.  You should see a message: "Your site is live at..."
4.  Click the link to view your deployed portfolio!

## Troubleshooting
-   **404 Error**: Ensure `vite.config.js` has `base: './'`. (We have already configured this).
-   **Images not loading**: Verify image paths are relative or use the `import` statement in React.
