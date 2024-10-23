def plot_circles_and_tangent(circle_start: Curve, circle_end: Curve, tangent: Line):
    # Plot circle_start
    centro_x1, centro_y1 = circle_start.center.p
    centro_x2, centro_y2 = circle_end.center.p
    radius = circle_start.radius

    # Circle points
    theta = np.linspace(0, 2 * np.pi, 100)
    x1 = centro_x1 + radius * np.cos(theta)
    y1 = centro_y1 + radius * np.sin(theta)

    x2 = centro_x2 + radius * np.cos(theta)
    y2 = centro_y2 + radius * np.sin(theta)

    x_line = [tangent.start_config.p[0], tangent.end_config.p[0]]
    y_line = [tangent.start_config.p[1], tangent.end_config.p[1]]

    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, label="Circonferenza 1")
    plt.plot(x2, y2, label="Circonferenza 2")
    plt.plot(x_line, y_line, label="Linea", color="red")

    # Imposta i limiti degli assi
    plt.xlim(-radius - 2, radius + 4)
    plt.ylim(-radius - 2, radius + 4)

    # Configura il grafico
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Due circonferenze e una linea")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.show()