import React, { useState, useEffect, useCallback, useRef } from 'react';
import { playClick, playMinesweeperExplosion, playMinesweeperWin } from '../../utils/soundEngine';

const DIFFICULTIES = {
    beginner: { rows: 9, cols: 9, mines: 10, name: 'Beginner' },
    intermediate: { rows: 16, cols: 16, mines: 40, name: 'Intermediate' }
};

const Minesweeper = () => {
    const [difficulty, setDifficulty] = useState('beginner');
    const { rows, cols, mines } = DIFFICULTIES[difficulty];

    const [board, setBoard] = useState([]);
    const [gameState, setGameState] = useState('ready'); // 'ready', 'playing', 'won', 'lost'
    const [faceStatus, setFaceStatus] = useState('smile'); // 'smile', 'scared', 'dead', 'cool'
    const [flagsLeft, setFlagsLeft] = useState(mines);
    const [timer, setTimer] = useState(0);
    const timerRef = useRef(null);

    const initBoard = useCallback((initialDifficulty = difficulty) => {
        const config = DIFFICULTIES[initialDifficulty];
        const newBoard = [];
        for (let r = 0; r < config.rows; r++) {
            const row = [];
            for (let c = 0; c < config.cols; c++) {
                row.push({
                    row: r,
                    col: c,
                    isMine: false,
                    isRevealed: false,
                    isFlagged: false,
                    neighborMines: 0
                });
            }
            newBoard.push(row);
        }

        if (timerRef.current) clearInterval(timerRef.current);
        setBoard(newBoard);
        setGameState('ready');
        setFaceStatus('smile');
        setFlagsLeft(config.mines);
        setTimer(0);
    }, [difficulty]);

    useEffect(() => {
        initBoard();
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, [initBoard]);

    const startTimer = () => {
        if (timerRef.current) clearInterval(timerRef.current);
        timerRef.current = setInterval(() => {
            setTimer(prev => Math.min(prev + 1, 999));
        }, 1000);
    };

    const populateMines = (clickedRow, clickedCol) => {
        const newBoard = board.map(r => r.map(c => ({ ...c })));
        let placed = 0;
        while (placed < mines) {
            const r = Math.floor(Math.random() * rows);
            const c = Math.floor(Math.random() * cols);
            // Don't place on first click or already mined cell
            if (!newBoard[r][c].isMine && !(Math.abs(r - clickedRow) <= 1 && Math.abs(c - clickedCol) <= 1)) {
                newBoard[r][c].isMine = true;
                placed++;
            }
        }

        // Count neighbors
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                if (!newBoard[r][c].isMine) {
                    let count = 0;
                    for (let dr = -1; dr <= 1; dr++) {
                        for (let dc = -1; dc <= 1; dc++) {
                            const nr = r + dr;
                            const nc = c + dc;
                            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && newBoard[nr][nc].isMine) {
                                count++;
                            }
                        }
                    }
                    newBoard[r][c].neighborMines = count;
                }
            }
        }
        return newBoard;
    };

    const revealCell = (r, c, currentBoard = board) => {
        let b = currentBoard;
        if (gameState === 'ready') {
            b = populateMines(r, c);
            setGameState('playing');
            startTimer();
        }

        const cell = b[r][c];
        if (cell.isRevealed || cell.isFlagged) return;

        playClick();

        if (cell.isMine) {
            // Lost!
            clearInterval(timerRef.current);
            setGameState('lost');
            setFaceStatus('dead');
            playMinesweeperExplosion();

            // Reveal all mines
            const revealedBoard = b.map(row => row.map(cellItem => ({
                ...cellItem,
                isRevealed: cellItem.isMine ? true : cellItem.isRevealed
            })));
            revealedBoard[r][c].isHit = true;
            setBoard(revealedBoard);
            return;
        }

        // Flood fill empty cells
        const updatedBoard = b.map(row => row.map(cellItem => ({ ...cellItem })));
        const stack = [[r, c]];
        updatedBoard[r][c].isRevealed = true;

        while (stack.length > 0) {
            const [currR, currC] = stack.pop();
            const currCell = updatedBoard[currR][currC];

            if (currCell.neighborMines === 0 && !currCell.isMine) {
                for (let dr = -1; dr <= 1; dr++) {
                    for (let dc = -1; dc <= 1; dc++) {
                        const nr = currR + dr;
                        const nc = currC + dc;
                        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                            const neighbor = updatedBoard[nr][nc];
                            if (!neighbor.isRevealed && !neighbor.isFlagged) {
                                neighbor.isRevealed = true;
                                if (neighbor.neighborMines === 0 && !neighbor.isMine) {
                                    stack.push([nr, nc]);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check Win Condition
        let unrevealedNonMines = 0;
        for (let rowIdx = 0; rowIdx < rows; rowIdx++) {
            for (let colIdx = 0; colIdx < cols; colIdx++) {
                if (!updatedBoard[rowIdx][colIdx].isMine && !updatedBoard[rowIdx][colIdx].isRevealed) {
                    unrevealedNonMines++;
                }
            }
        }

        if (unrevealedNonMines === 0) {
            clearInterval(timerRef.current);
            setGameState('won');
            setFaceStatus('cool');
            setFlagsLeft(0);
            playMinesweeperWin();
        }

        setBoard(updatedBoard);
    };

    const toggleFlag = (e, r, c) => {
        e.preventDefault();
        if (gameState === 'won' || gameState === 'lost') return;
        const cell = board[r][c];
        if (cell.isRevealed) return;

        playClick();
        const updatedBoard = board.map(row => row.map(cellItem => ({ ...cellItem })));
        const willBeFlagged = !cell.isFlagged;
        updatedBoard[r][c].isFlagged = willBeFlagged;
        setBoard(updatedBoard);
        setFlagsLeft(prev => willBeFlagged ? prev - 1 : prev + 1);
    };

    const getNumberColor = (num) => {
        switch (num) {
            case 1: return '#0000FF';
            case 2: return '#008000';
            case 3: return '#FF0000';
            case 4: return '#000080';
            case 5: return '#800000';
            case 6: return '#008080';
            case 7: return '#000000';
            case 8: return '#808080';
            default: return '#000000';
        }
    };

    const formatDigits = (num) => {
        const clamped = Math.max(-99, Math.min(999, num));
        if (clamped < 0) return '-' + String(Math.abs(clamped)).padStart(2, '0');
        return String(clamped).padStart(3, '0');
    };

    return (
        <div className="minesweeper-app">
            <div className="minesweeper-menu">
                <button 
                    className={`retro-btn-sm ${difficulty === 'beginner' ? 'active' : ''}`}
                    onClick={() => { setDifficulty('beginner'); initBoard('beginner'); }}
                >
                    Beginner (9x9)
                </button>
                <button 
                    className={`retro-btn-sm ${difficulty === 'intermediate' ? 'active' : ''}`}
                    onClick={() => { setDifficulty('intermediate'); initBoard('intermediate'); }}
                >
                    Intermediate (16x16)
                </button>
            </div>

            <div className="minesweeper-frame">
                {/* Header Bar */}
                <div className="minesweeper-header">
                    <div className="seven-segment-display">{formatDigits(flagsLeft)}</div>
                    <button 
                        className="minesweeper-face-btn"
                        onClick={() => initBoard(difficulty)}
                    >
                        {faceStatus === 'smile' && '🙂'}
                        {faceStatus === 'scared' && '😮'}
                        {faceStatus === 'dead' && '😵'}
                        {faceStatus === 'cool' && '😎'}
                    </button>
                    <div className="seven-segment-display">{formatDigits(timer)}</div>
                </div>

                {/* Game Grid */}
                <div 
                    className="minesweeper-grid"
                    style={{
                        gridTemplateColumns: `repeat(${cols}, 24px)`,
                        gridTemplateRows: `repeat(${rows}, 24px)`
                    }}
                    onMouseDown={() => gameState === 'playing' && setFaceStatus('scared')}
                    onMouseUp={() => gameState === 'playing' && setFaceStatus('smile')}
                >
                    {board.map((row, r) =>
                        row.map((cell, c) => {
                            let content = null;
                            if (cell.isRevealed) {
                                if (cell.isMine) {
                                    content = '💣';
                                } else if (cell.neighborMines > 0) {
                                    content = (
                                        <span style={{ color: getNumberColor(cell.neighborMines), fontWeight: 'bold' }}>
                                            {cell.neighborMines}
                                        </span>
                                    );
                                }
                            } else if (cell.isFlagged) {
                                content = '🚩';
                            }

                            return (
                                <div
                                    key={`${r}-${c}`}
                                    className={`minesweeper-cell ${cell.isRevealed ? 'revealed' : ''} ${cell.isHit ? 'mine-hit' : ''}`}
                                    onClick={() => revealCell(r, c)}
                                    onContextMenu={(e) => toggleFlag(e, r, c)}
                                >
                                    {content}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>
        </div>
    );
};

export default Minesweeper;
