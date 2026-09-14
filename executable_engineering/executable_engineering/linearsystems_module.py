import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pcolors

def visual_solve_2d(A, b, method='exact'):
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


def visualize_conditioning(A):
    """
    Visualizes the effect of the condition number by applying the matrix transformation
    and its inverse to the unit circle.
    
    Args:
        A: (2, 2) array-like matrix
    """
    A = np.array(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("visualize_conditioning requires a 2x2 matrix.")
        
    # Create the unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Apply transformation A to the unit circle
    transformed_A = A @ circle
    
    # Check if invertible, if so, apply A^-1 to the unit circle
    try:
        A_inv = np.linalg.inv(A)
        transformed_Ainv = A_inv @ circle
        cond = np.linalg.cond(A)
        subtitle = f"Condition Number: {cond:.2f}"
    except np.linalg.LinAlgError:
        transformed_Ainv = np.zeros_like(circle)
        subtitle = "Matrix is singular (Condition Number: ∞)"
        
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(f"Forward Transformation (A*x)", f"Inverse Transformation (A⁻¹*b)"),
        horizontal_spacing=0.1
    )
    
    # Subplot 1: Forward transformation
    fig.add_trace(go.Scatter(x=circle[0], y=circle[1], mode='lines', name='Unit Circle', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=transformed_A[0], y=transformed_A[1], mode='lines', name='A * Circle', line=dict(color='blue')), row=1, col=1)
    
    # Subplot 2: Inverse transformation
    fig.add_trace(go.Scatter(x=circle[0], y=circle[1], mode='lines', name='Unit Circle', line=dict(color='gray', dash='dash'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=transformed_Ainv[0], y=transformed_Ainv[1], mode='lines', name='A⁻¹ * Circle', line=dict(color='red')), row=1, col=2)
    
    # Update axes to be equal aspect ratio
    max_A = np.max(np.abs(transformed_A)) * 1.1 if np.max(np.abs(transformed_A)) > 0 else 1
    max_Ainv = np.max(np.abs(transformed_Ainv)) * 1.1 if np.max(np.abs(transformed_Ainv)) > 0 else 1
    
    # Ensure they are at least 1.1 so the unit circle is visible
    max_A = max(max_A, 1.1)
    max_Ainv = max(max_Ainv, 1.1)
    
    fig.update_xaxes(range=[-max_A, max_A], row=1, col=1)
    fig.update_yaxes(range=[-max_A, max_A], scaleanchor="x", scaleratio=1, row=1, col=1)
    
    fig.update_xaxes(range=[-max_Ainv, max_Ainv], row=1, col=2)
    fig.update_yaxes(range=[-max_Ainv, max_Ainv], scaleanchor="x2", scaleratio=1, row=1, col=2)
    
    fig.update_layout(
        title=f"Visualizing Matrix Distortion ({subtitle})",
        width=900,
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def visualize_matrix_norms(A):
    """
    Visualizes matrix norms by plotting the transformation of a unit circle
    and overlaying bounding circles based on different matrix norms.
    
    Args:
        A: (2, 2) array-like matrix
    """
    import numpy as np
    import plotly.graph_objects as go
    
    A = np.array(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("visualize_matrix_norms requires a 2x2 matrix.")
        
    # 1. Calculate the norms
    norm_2 = np.linalg.norm(A, ord=2)
    norm_F = np.linalg.norm(A, ord='fro')
    norm_inf = np.linalg.norm(A, ord=np.inf)
    
    # 2. Transform the unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle
    
    # 3. Create circles for the norms
    circle_2 = norm_2 * circle
    circle_F = norm_F * circle
    circle_inf = norm_inf * circle
    
    # 4. Plot!
    fig = go.Figure()
    
    # Plot the transformed ellipse
    fig.add_trace(go.Scatter(x=ellipse[0], y=ellipse[1], mode='lines', name='A * Unit Circle (Ellipse)', line=dict(color='blue', width=3)))
    
    # Plot the norm bounding circles
    fig.add_trace(go.Scatter(x=circle_2[0], y=circle_2[1], mode='lines', name=f'2-Norm (Radius {norm_2:.2f})', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=circle_F[0], y=circle_F[1], mode='lines', name=f'Frobenius Norm (Radius {norm_F:.2f})', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=circle_inf[0], y=circle_inf[1], mode='lines', name=f'Infinity Norm (Radius {norm_inf:.2f})', line=dict(color='red', dash='dash')))
    
    # Setup axes
    max_val = norm_inf * 1.1
    fig.update_xaxes(range=[-max_val, max_val])
    fig.update_yaxes(range=[-max_val, max_val], scaleanchor="x", scaleratio=1)
    fig.update_layout(title="Matrix Norms as Bounds on Geometric Distortion", width=700, height=700)
    
    return fig

def jacobi_step(A, b, x):
    """Performs a single Jacobi iteration step."""
    n = len(x)
    x_new = np.zeros_like(x)
    for i in range(n):
        x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x_new

def gauss_seidel_step(A, b, x, omega = 1):
    """Performs a single Gauss-Seidel iteration step."""
    n = len(x)
    for i in range(n):
        x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x

def sor_step(A, b, x, omega=1.05):
    """Performs a single SOR iteration step."""
    n = len(x)
    for i in range(n):
        x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:]))
    return x

def visualize_convergence_2d(A, b, iterations, surface_type='residual'):
    """
    Visualizes the convergence path of an iterative solver over a 2D contour surface.
    
    Args:
        A: (2, 2) coefficient matrix.
        b: (2,) right-hand side vector.
        iterations: List or array of (2,) vectors representing the guess at each step.
        surface_type: 'residual' plots ||Ax - b||. 'quadratic' plots 0.5 x^T A x - b^T x.
    """
    import numpy as np
    import plotly.graph_objects as go
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()
    iterations = np.array(iterations)
    
    if A.shape != (2, 2):
        raise ValueError("visualize_convergence_2d only supports 2D systems.")
        
    try:
        x_true = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x_true = np.zeros(2)
        
    # Determine bounds
    all_points = np.vstack([iterations, x_true])
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    
    # Add padding
    pad_x = max((max_x - min_x) * 0.3, 0.5)
    pad_y = max((max_y - min_y) * 0.3, 0.5)
    
    x_range = np.linspace(min_x - pad_x, max_x + pad_x, 100)
    y_range = np.linspace(min_y - pad_y, max_y + pad_y, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    if surface_type == 'residual':
        # ||Ax - b||
        Z = np.sqrt((A[0,0]*X + A[0,1]*Y - b[0])**2 + (A[1,0]*X + A[1,1]*Y - b[1])**2)
        title = "Convergence Path over Residual Surface ||Ax - b||"
    elif surface_type == 'quadratic':
        # 0.5 x^T A x - b^T x
        Z = 0.5 * (A[0,0]*X**2 + (A[0,1] + A[1,0])*X*Y + A[1,1]*Y**2) - (b[0]*X + b[1]*Y)
        title = "Convergence Path over Quadratic Surface f(x)"
    else:
        raise ValueError("surface_type must be 'residual' or 'quadratic'")
        
    fig = go.Figure()
    
    # Contour plot
    fig.add_trace(go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale='Viridis',
        contours=dict(showlabels=True, labelfont=dict(size=12, color='white')),
        showscale=False,
        opacity=0.7
    ))
    
    # Convergence path
    import numpy as np
    import plotly.express as px
    
    # We want to draw each segment with a color from the colormap
    if len(iterations) > 1:
        colors = px.colors.sample_colorscale('magma', np.linspace(0, 1, len(iterations)-1))
        for i in range(len(iterations)-1):
            fig.add_trace(go.Scatter(
                x=iterations[i:i+2, 0], y=iterations[i:i+2, 1],
                mode='lines',
                line=dict(color=colors[i], width=3),
                showlegend=False
            ))
        
        # Add a dummy trace just to show the colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                size=0, 
                color=[0, len(iterations)-1], 
                colorscale='magma', 
                showscale=True,
                colorbar=dict(title="Iteration", len=0.5, y=0.5, x=1.1)
            ),
            showlegend=False
        ))
    else:
        # Just in case there is no path
        fig.add_trace(go.Scatter(
            x=iterations[:, 0], y=iterations[:, 1],
            mode='markers',
            marker=dict(size=8, color='white'),
            name='Iterations'
        ))
    

    
    fig.update_layout(
        title=title,
        width=700, height=600,
        xaxis_title="x₁",
        yaxis_title="x₂",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_diagonally_dominant_matrix(n):
    """Creates a diagonally dominant random matrix.

    Args:
        n: The size of the matrix.

    Returns:
        A diagonally dominant random matrix.
    """
    import numpy as np
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + np.random.rand()
    return A

def iter_solve(A, b, method, tol=1e-6, track_history=False):
    """
    General wrapper function that repeatedly applies an iterative step 
    until the residual error falls below a tolerance.
    """
    import numpy as np
    x = np.zeros(len(b))
    max_iter = 1000
    
    if track_history:
        iterations = [x.copy()]

    for i in range(max_iter):
        x_new = method(A, b, x.copy())
        if track_history:
            iterations.append(x_new.copy())
            
        if np.linalg.norm(A @ x_new - b) < tol:
            print(f'Converged after {i+1} iterations.')
            break
        x = x_new.copy()
        
    if track_history:
        return x, iterations
    return x

def steepest_descent(A, b, x0, tol=1e-6, max_iter=100, track_history=False):
    """Solves a linear system Ax = b using the method of steepest descent."""
    import numpy as np
    x = x0.copy()
    if track_history:
        iterations = [x.copy()]
        
    for i in range(max_iter):
        r = A @ x - b
        if np.linalg.norm(r) < tol:
            if track_history: print(f'Converged after {i+1} iterations.')
            break
        alpha = np.dot(r, r) / np.dot(r, A @ r)
        x = x - alpha * r
        if track_history:
            iterations.append(x.copy())
            
    if track_history:
        return x, iterations
    return x

def conjugate_gradient(A, b, x0, tol=1e-6, track_history=False):
    """Wrapper around SciPy CG to track iterations if needed."""
    import numpy as np
    from scipy.sparse.linalg import cg
    
    if track_history:
        iterations = [x0.copy()]
        def callback(xk):
            iterations.append(np.copy(xk))
        solution, info = cg(A, b, x0=x0, tol=tol, callback=callback)
        return solution, np.array(iterations)
    else:
        solution, info = cg(A, b, x0=x0, tol=tol)
        return solution

class IterationTracker:
    """
    A clean callback tracker for native SciPy solvers (cg, gmres, etc.).
    """
    def __init__(self, x0=None):
        import numpy as np
        if x0 is not None:
            self.iterations = [np.array(x0).copy()]
        else:
            self.iterations = []

    def __call__(self, xk):
        import numpy as np
        self.iterations.append(np.array(xk).copy())
