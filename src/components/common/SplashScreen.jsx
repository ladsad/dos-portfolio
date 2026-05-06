import React, { useState, useEffect } from 'react';

const SplashScreen = ({ onComplete, onEnterBios }) => {
    const [lines, setLines] = useState([]);
    const [memory, setMemory] = useState(0);

    useEffect(() => {
        // Memory counter animation
        const memInterval = setInterval(() => {
            setMemory(prev => {
                if (prev >= 640) {
                    clearInterval(memInterval);
                    return 640;
                }
                return prev + 16;
            });
        }, 50);

        // Text sequence
        const sequence = [
            { text: "BIOS Date 01/01/90 15:22:34 Ver: 1.02", delay: 100 },
            { text: "CPU: NEC V20, Speed: 8 MHz", delay: 600 },
            { text: "640K RAM System... OK", delay: 1500 },
            { text: "Detecting Primary Master... SALAD-DRIVE-C", delay: 2000 },
            { text: "Detecting Primary Slave... None", delay: 2500 },
            { text: "", delay: 3000 },
            { text: "Booting from C:...", delay: 3500 },
            { text: "Starting MS-DOS...", delay: 4500 },
        ];

        let timeouts = [];

        sequence.forEach(({ text, delay }) => {
            const timeout = setTimeout(() => {
                setLines(prev => [...prev, text]);
            }, delay);
            timeouts.push(timeout);
        });

        const finishTimeout = setTimeout(() => {
            onComplete();
        }, 5500);
        timeouts.push(finishTimeout);

        const handleKeyDown = (e) => {
            if (e.key === 'Delete' || e.key === 'Del') {
                if (onEnterBios) {
                    onEnterBios();
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        return () => {
            clearInterval(memInterval);
            timeouts.forEach(clearTimeout);
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [onComplete, onEnterBios]);

    return (
        <div className="splash-screen" onClick={onComplete}>
            <div className="bios-text">
                <div>Award Modular BIOS v4.51PG, An Energy Star Ally</div>
                <div>Copyright (C) 1984-90, Award Software, Inc.</div>
                <br />
                <div>PENTIUM-II CPU at 400MHz</div>
                <div>Memory Test : {memory}K OK</div>
                <br />
                {lines.map((line, i) => (
                    <div key={i}>{line}</div>
                ))}
                <div className="cursor-blink">_</div>
                <br />
                <div className="bios-footer">
                    Press DEL to enter SETUP
                    <br />
                    01/01/90-i430VX-2A59Gt2BC-00
                </div>
            </div>
        </div>
    );
};

export default SplashScreen;
