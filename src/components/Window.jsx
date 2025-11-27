import React, { useRef, useCallback, useState, useEffect } from 'react';
// Window component with manual resize and auto-height support
import Draggable from 'react-draggable';
import { X, Minus, Square } from 'lucide-react';

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
    const resizeRef = useRef(null);

    // Initialize size if provided (though we default to 600x400 for now)
    // We could pass initialSize prop if needed later.

    const handleResizeMove = useCallback((e) => {
        if (!resizeRef.current) return;
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
    }, []);

    const handleResizeEnd = useCallback(() => {
        setIsResizing(false);
        resizeRef.current = null;
        window.removeEventListener('mousemove', handleResizeMove);
        window.removeEventListener('mouseup', handleResizeEnd);
    }, [handleResizeMove]);

    const handleResizeStart = (e, direction) => {
        e.preventDefault();
        e.stopPropagation();
        setIsResizing(true);

        // Capture current dimensions from DOM to handle 'auto' height
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

    // Cleanup
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
            onMouseDown={onFocus}

            nodeRef={nodeRef}
            bounds="parent"
            disabled={isResizing || isMobile} // Disable dragging while resizing or on mobile
        >
            <div
                ref={nodeRef}
                className={`window-frame ${isActive ? 'active' : ''} ${className}`}
                style={{
                    zIndex,
                    position: 'absolute',
                    width: isMobile ? '100%' : size.width,
                    height: isMobile ? 'calc(100% - 40px)' : size.height,
                    maxHeight: isMobile ? 'none' : (size.height === 'auto' ? '80vh' : 'none') // Use 80vh for auto, none for manual
                }}
                onClick={onFocus}
            >
                <div className="window-header">
                    <span className="window-title">{title}</span>
                    <div className="window-controls">
                        <button onClick={(e) => { e.stopPropagation(); onMinimize(id); }} className="control-btn">
                            <Minus size={14} />
                        </button>
                        <button className="control-btn" disabled>
                            <Square size={12} />
                        </button>
                        <button onClick={(e) => { e.stopPropagation(); onClose(id); }} className="control-btn close-btn">
                            <X size={14} />
                        </button>
                    </div>
                </div>
                <div className="window-content">
                    {children}
                </div>

                {/* Resize Handles */}
                {!isMobile && (
                    <>
                        <div className="resize-handle resize-handle-r" onMouseDown={(e) => handleResizeStart(e, 'r')} />
                        <div className="resize-handle resize-handle-b" onMouseDown={(e) => handleResizeStart(e, 'b')} />
                        <div className="resize-handle resize-handle-br" onMouseDown={(e) => handleResizeStart(e, 'br')} />
                    </>
                )}
            </div>
        </Draggable>
    );
};

export default Window;
