# DOS-Style Portfolio Website

A retro-themed personal portfolio website inspired by Windows 98 and MS-DOS, built with React and Vite.

**Live Demo:** [https://ladsad.github.io/dos-portfolio/](https://ladsad.github.io/dos-portfolio/)

## Features

-   **Interactive Terminal**: A fully functional command-line interface with custom commands (`about`, `projects`, `skills`, `open <project>`).
-   **Windows 98 GUI**: Authentic desktop experience with a Start Menu, Taskbar, and System Tray.
-   **Window Management**: Draggable, resizable, minimizable, and focusable windows.
-   **Project Dashboard**: Detailed project views with ASCII art headers and rich text descriptions.
-   **Resume Viewer**: A dedicated "paper-style" PDF viewer for showcasing your CV.
-   **Mobile Responsive**: Adaptive layout that works seamlessly on phones and tablets.
-   **Easter Eggs**: BIOS boot sequence, "Press DEL for Setup", and more.

## Tech Stack

-   **Frontend**: React 19, Vite
-   **Styling**: Vanilla CSS (No frameworks, pure retro styling)
-   **Libraries**:
    -   `react-draggable`: For window management.
    -   `lucide-react`: For icons.
    -   `react-markdown`: For rendering project content.

## Project Structure

```
src/
├── components/         # React components
│   ├── BiosSetup.jsx   # Easter egg: BIOS setup screen
│   ├── StartMenu.jsx   # Windows 98 Start Menu logic
│   ├── Taskbar.jsx     # Taskbar with clock and window list
│   ├── Terminal.jsx    # Interactive command-line interface
│   ├── Window.jsx      # Draggable/Resizable window wrapper
│   └── ...
├── data/
│   └── portfolio.js    # Centralized content (projects, resume, etc.)
├── App.jsx             # Main application layout
└── index.css           # Global retro styles (Vanilla CSS)
```

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ladsad/dos-portfolio.git
    cd dos-portfolio
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Run the development server**:
    ```bash
    npm run dev
    ```

4.  **Build for production**:
    ```bash
    npm run build
    ```

## Deployment

This project is configured for deployment on **GitHub Pages**.

For detailed deployment instructions, please refer to [DEPLOYMENT.md](./DEPLOYMENT.md).

## License

This project is open source and available under the [MIT License](LICENSE).
