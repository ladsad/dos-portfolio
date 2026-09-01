import React, { useRef, useCallback, useState, useEffect } from 'react';
import Draggable from 'react-draggable';
import { X, Minus, Square, Copy } from 'lucide-react';
import { playClick } from '../../utils/soundEngine';

const Window = ({
    id,
    title,
    children,
    onClose,
    onMinimize,
    isActive,
    onFocus,
    zIndex,
    initialPosition = { x: 0, y: 0 },
    initialSize,
    className = '',
    isMobile = false
}) => {
    const nodeRef = useRef(null);
    const [size, setSize] = useState(initialSize || { width: 600, height: 'auto' });
    const [isResizing, setIsResizing] = useState(false);
    const [isMaximized, setIsMaximized] = useState(false);
    const resizeRef = useRef(null);

    const toggleMaximize = useCallback((e) => {
        if (e) {
            e.stopPropagation();
        }
        if (isMobile) return;
        playClick();
        setIsMaximized(prev => !prev);
    }, [isMobile]);

    const handleResizeMove = useCallback((e) => {
        if (!resizeRef.current || isMaximized) return;
        const { startX, startY, startWidth, startHeight, direction } = resizeRef.current;

        let newWidth = startWidth;
        let newHeight = startHeight;

        if (direction.includes('r')) {
            newWidth = Math.max(300, startWidth + (e.clientX - startX));
            newWidth = Math.min(newWidth, window.innerWidth - 20);
        }
        if (direction.includes('b')) {
            newHeight = Math.max(200, startHeight + (e.clientY - startY));
            newHeight = Math.min(newHeight, window.innerHeight - 40);
        }

        setSize({ width: newWidth, height: newHeight });
    }, [isMaximized]);

    const handleResizeEnd = useCallback(() => {
        setIsResizing(false);
        resizeRef.current = null;
        window.removeEventListener('mousemove', handleResizeMove);
        window.removeEventListener('mouseup', handleResizeEnd);
    }, [handleResizeMove]);

    const handleResizeStart = (e, direction) => {
        if (isMaximized) return;
        e.preventDefault();
        e.stopPropagation();
        setIsResizing(true);

        const currentWidth = nodeRef.current ? nodeRef.current.offsetWidth : size.width;
        const currentHeight = nodeRef.current ? nodeRef.current.offsetHeight : size.height;

        resizeRef.current = {
            startX: e.clientX,
            startY: e.clientY,
            startWidth: currentWidth,
            startHeight: currentHeight,
            direction
        };
        window.addEventListener('mousemove', handleResizeMove);
        window.addEventListener('mouseup', handleResizeEnd);
    };

    useEffect(() => {
        return () => {
            window.removeEventListener('mousemove', handleResizeMove);
            window.removeEventListener('mouseup', handleResizeEnd);
        };
    }, [handleResizeMove, handleResizeEnd]);

    return (
        <Draggable
            handle=".window-header"
            defaultPosition={initialPosition}
            position={isMaximized ? { x: 0, y: 0 } : undefined}
            onMouseDown={onFocus}
            nodeRef={nodeRef}
            bounds="parent"
            disabled={isResizing || isMobile || isMaximized}
        >
            <div
                ref={nodeRef}
                className={`window-frame-outer ${isActive ? 'active' : ''} ${isMaximized ? 'maximized' : ''} ${className}`}
                style={{
                    zIndex,
                    position: isMaximized ? 'fixed' : 'absolute',
                    top: isMaximized ? 0 : undefined,
                    left: isMaximized ? 0 : undefined,
                    width: isMobile || isMaximized ? '100%' : size.width,
                    height: isMobile || isMaximized ? 'calc(100vh - 40px)' : size.height,
                    maxHeight: isMobile || isMaximized ? 'none' : (size.height === 'auto' ? '80vh' : 'none')
                }}
                onClick={onFocus}
            >
                <div className={`window-frame ${isActive ? 'active' : ''}`} style={{ width: '100%', height: '100%' }}>
                    <div className="window-header" onDoubleClick={toggleMaximize}>
                        <span className="window-title">{title}</span>
                        <div className="window-controls">
                            <button onClick={(e) => { e.stopPropagation(); onMinimize(id); }} className="control-btn" title="Minimize">
                                <Minus size={14} />
                            </button>
                            <button onClick={toggleMaximize} className="control-btn" title={isMaximized ? "Restore" : "Maximize"}>
                                {isMaximized ? <Copy size={11} /> : <Square size={12} />}
                            </button>
                            <button onClick={(e) => { e.stopPropagation(); onClose(id); }} className="control-btn close-btn" title="Close">
                                <X size={14} />
                            </button>
                        </div>
                    </div>
                    <div className="window-content">
                        {children}
                    </div>

                    {!isMobile && !isMaximized && (
                        <>
                            <div className="resize-handle resize-handle-r" onMouseDown={(e) => handleResizeStart(e, 'r')} />
                            <div className="resize-handle resize-handle-b" onMouseDown={(e) => handleResizeStart(e, 'b')} />
                            <div className="resize-handle resize-handle-br" onMouseDown={(e) => handleResizeStart(e, 'br')} />
                        </>
                    )}
                </div>
            </div>
        </Draggable>
    );
};

export default Window;
