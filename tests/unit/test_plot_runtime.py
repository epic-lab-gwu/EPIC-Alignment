from vicon_ws.viz import plot_runtime


def test_should_enable_interactive_plot_explicit_true() -> None:
    assert plot_runtime.should_enable_interactive_plot(plot=False, plot_interactive=True) is True


def test_should_enable_interactive_plot_plot_false() -> None:
    assert plot_runtime.should_enable_interactive_plot(plot=False, plot_interactive=False) is False
