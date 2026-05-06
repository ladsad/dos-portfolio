import React, { useState } from 'react';
import Layout from './components/common/Layout';
import WindowManager from './components/os/WindowManager';
import SplashScreen from './components/common/SplashScreen';
import BiosSetup from './components/apps/BiosSetup';
import './index.css';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [showBios, setShowBios] = useState(false);
  const [theme] = useState(() => localStorage.getItem('displayMode') || 'retro');

  const handleEnterBios = () => {
    setShowBios(true);
    setIsLoading(false);
  };

  const handleExitBios = () => {
    setShowBios(false);
    setIsLoading(true);
  };

  if (showBios) {
    return <BiosSetup onExit={handleExitBios} />;
  }

  return (
    <>
      {isLoading ? (
        <SplashScreen
          onComplete={() => setIsLoading(false)}
          onEnterBios={handleEnterBios}
        />
      ) : (
        <Layout theme={theme}>
          <WindowManager />
        </Layout>
      )}
    </>
  );
}

export default App;
