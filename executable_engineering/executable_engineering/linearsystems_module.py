import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors

def visual_solve_2d(A, b, method='pseudo'):
    """
    Solves and visualizes a 2D linear system Ax = b.
    Plots the lines represented by the rows of A and b, and the solution point.
    
    Args:
        A: (N, 2) array-like of coefficients.
        b: (N,) array-like of constants.
        method: 'exact' (uses np.linalg.solve) or 'pseudo' (uses np.linalg.pinv).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    
    if A.shape[1] != 2:
        raise ValueError("visual_solve_2d only supports 2D systems (A must have 2 columns).")
    
    # 1. Solve the system based on chosen method
    sol = None
    plot_star = False
    
    if method == 'exact':
        try:
            if A.shape[0] == A.shape[1]:
                sol = np.linalg.solve(A, b)
                if np.allclose(A @ sol, b):
                    plot_star = True
                    sol_label = "Exact Solution"
        except np.linalg.LinAlgError:
            pass # No exact unique solution
            
    elif method == 'pseudo':
        sol = np.linalg.pinv(A) @ b
        plot_star = True
        
        if np.allclose(A @ sol, b):
            sol_label = "Exact Solution (via Pseudoinverse)"
        else:
            sol_label = "Best Fit Solution"
    else:
        raise ValueError("method must be 'exact' or 'pseudo'")
        
    # 2. Determine plotting range
    if sol is not None:
        center_x = sol[0]
        center_y = sol[1]
    else:
        # Fallback if no solution is found
        center_x = 10
        center_y = 10
        
    x_margin = 10
    x_range = (center_x - x_margin, center_x + x_margin)
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    
    fig = go.Figure()
    colors = pcolors.qualitative.Plotly
    
    # 3. Plot each line
    for i in range(A.shape[0]):
        a1, a2 = A[i]
        bi = b[i]
        line_name = f"Row {i+1}"
        
        if np.abs(a2) > 1e-10:
            y_vals = (bi - a1 * x_vals) / a2
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=line_name, line=dict(color=colors[i % len(colors)])))
        else:
            if np.abs(a1) > 1e-10:
                x_val = bi / a1
                fig.add_trace(go.Scatter(x=[x_val, x_val], y=[center_y - 100, center_y + 100], mode='lines', name=line_name, line=dict(color=colors[i % len(colors)])))
                
    # 4. Plot solution point if required
    if plot_star and sol is not None:
        hover_text = f"x={sol[0]:.4g}, y={sol[1]:.4g}"
        fig.add_trace(go.Scatter(
            x=[sol[0]], 
            y=[sol[1]], 
            mode='markers', 
            marker=dict(color='black', size=12, symbol='star'),
            name=sol_label,
            hoverinfo='text',
            hovertext=hover_text
        ))
    
    # 5. Auto-adjust y-range nicely around the center
    y_margin = (x_range[1] - x_range[0]) / 2.0
    yaxis_dict = dict(range=[center_y - y_margin, center_y + y_margin], scaleanchor="x", scaleratio=1)
        
    fig.update_layout(
        xaxis=dict(range=[x_range[0], x_range[1]]),
        yaxis=yaxis_dict,
        width=700,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig
