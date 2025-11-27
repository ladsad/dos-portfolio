import React, { useState } from 'react';
import Layout from './components/Layout';
import WindowManager from './components/WindowManager';
import SplashScreen from './components/SplashScreen';
import BiosSetup from './components/BiosSetup';
import './index.css';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [showBios, setShowBios] = useState(false);

  const handleEnterBios = () => {
    setShowBios(true);
    setIsLoading(false);
  };

  const handleExitBios = () => {
    setShowBios(false);
    setIsLoading(true); // Restart boot sequence or just go to OS? Let's restart boot for realism
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
        <Layout>
          <WindowManager />
        </Layout>
      )}
    </>
  );
}

export default App;
