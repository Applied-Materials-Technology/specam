import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from specam.constants import abs0

from specam.data import SpectralData


class SpectrumPlot():
    def __init__(self, plot_data_kwargs: list[dict]=None, i_T=None):
        self.lines = []  
        self.line_true = None
        self.points_true = None 

        plot_slider = False
        if i_T is None:
            i_T = 0
            plot_slider = True

        fig = plt.figure(figsize=(4, 3), constrained_layout=(not plot_slider))
        ax = plt.gca()

        ax.set_xlabel("Wavelength ($\mu$m)")
        ax.set_ylabel("Intensity")
        ax.legend()

        for kwargs in plot_data_kwargs or []:
            self.add_data(**kwargs)

        if plot_slider:
            def slider_key(event, slider):
                if event.key == 'left':
                    slider.set_val(slider.val - 1)
                elif event.key == 'right':
                    slider.set_val(slider.val + 1)

            slider_max = i_T
            
            fig.subplots_adjust(bottom=0.2)
            i_slider = Slider(
                ax=fig.add_axes([0.18, 0.02, 0.65, 0.01]),
                label="",
                valmin=0,
                # valmax=len(test_data["I"]) - 1,
                # valmax=i_T,
                valmax=10,
                valstep=1,
                valinit=i_T,
            )
            i_slider.on_changed(self.update)
            fig.canvas.mpl_connect(
                'key_press_event', lambda e: slider_key(e, i_slider)
            )
            self.i_slider = i_slider  # keep fig alive
            # rtn = rtn + (i_slider ,)

        self.fig = fig
        self.ax = ax

    def check_batch_size(self, batch_size):
        if len(self.lines) == 0:
            return
        
        plot_data = self.lines[0]['plot_data']
        assert plot_data.batch_size == batch_size

    def add_data(self, kind=None, **kwargs):
        func = {
            'true': self.add_true,
        }.get(str(kind).lower(), self.add_result)
        return func(**kwargs)

    def add_result(self, plot_data: type[SpectralData], label: str, **kwargs):
        self.check_batch_size(plot_data.batch_size)
        mpl_line = self.ax.plot([0], [0], label=label, **kwargs)[0]
        line = {
            'plot_data': plot_data,
            'label': label,
            'mpl_line': mpl_line,
        }
        if len(self.lines) == 0 and self.i_slider is not None:
            self.i_slider.valmax = plot_data.batch_size
            self.i_slider.ax.set_xlim(
                self.i_slider.valmin, 
                self.i_slider.valmax
            )
        self.lines.append(line)
        return line

    def add_true(self, plot_data: type[SpectralData], label: str = 'True'):
        line = self.add_result(plot_data, label, alpha=0.5)
        line['mpl_points'] = self.ax.scatter([0], [0], marker=".", alpha=0.5)
        self.line_true_idx = len(self.lines) - 1
        return line
    
    def update_line(self, i, plot_data, mpl_line, **kwargs):
        lam = plot_data.lam * 1e6
        intensity = plot_data.get_intensity(i)
        mpl_line.set_xdata(lam)
        mpl_line.set_ydata(intensity)

        if 'mpl_points' in kwargs:
            kwargs['mpl_points'].set_offsets(
                np.hstack((lam[:, None], intensity[:, None]))
            )

        # if name == 'ratio':
            #     # print(len(lam))
            #     # print(len(results["I"][:, i_T]))
            #     line_ratio.set_xdata(lam * 1e6)
            #     line_ratio.set_ydata(results["I"][:, i_T])

    def update(self, i_T):
        for line in self.lines:
            self.update_line(i_T, **line)

        # # func = intensity_func_log if log else intensity_func
        # # func_fit = intensity_func_log_fit if log else intensity_func_fit

        # # func = test_data[intensity_func_name]
        # lam = test_data['lam']

        # # I_true = func(
        # #     lam,
        # #     self.test_data["T"][i_T],
        # #     lam_0=self.props["lam_0"],
        # #     lam_inf=self.props["lam_inf"],
        # #     **self.test_data["intensity_params"],
        # # )
        # I_true = test_data['I'][i_T]


        # line_true.set_xdata(lam * 1e6)
        # line_true.set_ydata(I_true)
        # y_vals = I_true[:, None]
        # if log:
        #     y_vals = np.log(y_vals)
        # points.set_offsets(
        #     np.hstack((lam[:, None] * 1e6, y_vals))
        # )

            
        # title = f"{test_data['T'][i_T] - abs0:.0f}$^\circ$C"
        # if len(results) == 1:
        #     result = results[plot_result_names[0]]
        #     temp = results['T'][i_T] - abs0
        #     C = results['C'][i_T]
        #     D = results['D'][i_T]
        #     title += f" {temp:.0f}"
        #     for param_name in results['intensity_params']:
        #         title += f" {param_name} {results[param_name][i_T]:.3f}"
        # title += " $^\circ$C"

        # ax.set_title(title)
        # for ax_i in [ax, ax_diff]:
        for ax_i in [self.ax,]:
            if ax_i is None:
                continue
            ax_i.relim()
            ax_i.autoscale_view()
            # if ylim is not None:
            #     ax_i.set_ylim(ylim)

        self.fig.canvas.draw_idle()


        


# removed: manual_fit, plot_diff
def plot_spectrum(
    plot_data: dict[str, type[SpectralData]], 

    test_data,  # as in camera.test_data
    results, # dict of dict
    i_T=None,
    # plot_result_names=None,
    log=False,
    ylim=None,
    # intensity_func=None,
    # intensity_func_log=None,
):
    # test_data, uses 'I', 'lam'
    # results["props"], 'lam_0', 'lam_inf' 

    # if isinstance(plot_result_names, str):
    #     plot_result_names = [plot_result_names]
    # plot_result_names = plot_result_names or []

    intensity_func_name = "intensity_func_log" if log else "intensity_func"

    def update(i_T):
        # func = intensity_func_log if log else intensity_func
        # func_fit = intensity_func_log_fit if log else intensity_func_fit

        # func = test_data[intensity_func_name]
        lam = test_data['lam']

        # I_true = func(
        #     lam,
        #     self.test_data["T"][i_T],
        #     lam_0=self.props["lam_0"],
        #     lam_inf=self.props["lam_inf"],
        #     **self.test_data["intensity_params"],
        # )
        I_true = test_data['I'][i_T]


        line_true.set_xdata(lam * 1e6)
        line_true.set_ydata(I_true)
        y_vals = I_true[:, None]
        if log:
            y_vals = np.log(y_vals)
        points.set_offsets(
            np.hstack((lam[:, None] * 1e6, y_vals))
        )
        
        for result_name, result in results.items():
            func_fit = result[intensity_func_name]
            params = (result[k][i_T] for k in result["intensity_params"])
            I_pred = func_fit(
                lam,
                result["T"][i_T],
                *params,
                # self.props["lam_0"],
                # self.props["lam_inf"],
                result["props"]["lam_0"],
                result["props"]["lam_inf"],
            )
            line_pred[result_name].set_xdata(lam * 1e6)
            line_pred[result_name].set_ydata(I_pred)

            # if name == 'ratio':
            #     # print(len(lam))
            #     # print(len(results["I"][:, i_T]))
            #     line_ratio.set_xdata(lam * 1e6)
            #     line_ratio.set_ydata(results["I"][:, i_T])

        # title = f"{test_data['T'][i_T] - abs0:.0f}$^\circ$C"
        # if len(results) == 1:
        #     result = results[plot_result_names[0]]
        #     temp = results['T'][i_T] - abs0
        #     C = results['C'][i_T]
        #     D = results['D'][i_T]
        #     title += f" {temp:.0f}"
        #     for param_name in results['intensity_params']:
        #         title += f" {param_name} {results[param_name][i_T]:.3f}"
        # title += " $^\circ$C"

        # ax.set_title(title)
        # for ax_i in [ax, ax_diff]:
        for ax_i in [ax,]:
            if ax_i is None:
                continue
            ax_i.relim()
            ax_i.autoscale_view()
            if ylim is not None:
                ax_i.set_ylim(ylim)

        fig.canvas.draw_idle()


    plot_slider = False
    if i_T is None:
        i_T = 0
        plot_slider = True

    fig = plt.figure(figsize=(4, 3), constrained_layout=(not plot_slider))
    ax = plt.gca()

    lines = {}
    for name, data in plot_data.items():
        (lines[name],) = ax.plot([0], [0], label=name)



    (line_true,) = ax.plot([0], [0], alpha=0.5)
    points = ax.scatter([0], [0], marker=".", alpha=0.5)

    line_pred = {}
    for result_name in results.keys():
        (line_pred[result_name],) = ax.plot([0], [0], label=result_name)


    # (line_ratio,) = ax.plot([0], [0], label='ratio_extra')


    ax.set_xlabel("Wavelength ($\mu$m)")
    ax.set_ylabel("Intensity")
    ax.legend()
        
    # update(i_T)

    rtn = ()

    if plot_slider:
        def slider_key(event, slider):
            if event.key == 'left':
                slider.set_val(slider.val - 1)
            elif event.key == 'right':
                slider.set_val(slider.val + 1)

        fig.subplots_adjust(bottom=0.2)
        ax_i = fig.add_axes([0.18, 0.02, 0.65, 0.01])
        i_slider = Slider(
            ax=ax_i,
            label="",
            valmin=0,
            valmax=len(test_data["I"]) - 1,
            valstep=1,
            valinit=i_T,
        )
        i_slider.on_changed(update)
        fig.canvas.mpl_connect('key_press_event', lambda e: slider_key(e, i_slider))
        # self.fig_widget = i_slider  # keep fig alive
        rtn = rtn + (i_slider ,)
    

    update(i_T)

    plt.show()

    return rtn

# def plot_spectrum(
#     test_data,  # as in camera.test_data
#     results, # dict of dict
#     i_T=None,
#     # plot_result_names=None,
#     log=False,
#     ylim=None,
#     # intensity_func=None,
#     # intensity_func_log=None,
#     plot_diff=False,
#     manual_fit=False,
# ):
#     # test_data, uses 'I', 'lam'
#     # results["props"], 'lam_0', 'lam_inf' 

#     # if isinstance(plot_result_names, str):
#     #     plot_result_names = [plot_result_names]
#     # plot_result_names = plot_result_names or []

#     intensity_func_name = "intensity_func_log" if log else "intensity_func"

#     def update(i_T):
#         # func = intensity_func_log if log else intensity_func
#         # func_fit = intensity_func_log_fit if log else intensity_func_fit

#         # func = test_data[intensity_func_name]
#         lam = test_data['lam']

#         # I_true = func(
#         #     lam,
#         #     self.test_data["T"][i_T],
#         #     lam_0=self.props["lam_0"],
#         #     lam_inf=self.props["lam_inf"],
#         #     **self.test_data["intensity_params"],
#         # )
#         I_true = test_data['I'][i_T]


#         line_true.set_xdata(lam * 1e6)
#         line_true.set_ydata(I_true)
#         y_vals = I_true[:, None]
#         if log:
#             y_vals = np.log(y_vals)
#         points.set_offsets(
#             np.hstack((lam[:, None] * 1e6, y_vals))
#         )
        
#         for result_name, result in results.items():
#             func_fit = result[intensity_func_name]
#             params = (result[k][i_T] for k in result["intensity_params"])
#             I_pred = func_fit(
#                 lam,
#                 result["T"][i_T],
#                 *params,
#                 # self.props["lam_0"],
#                 # self.props["lam_inf"],
#                 result["props"]["lam_0"],
#                 result["props"]["lam_inf"],
#             )
#             line_pred[result_name].set_xdata(lam * 1e6)
#             line_pred[result_name].set_ydata(I_pred)

#             if plot_diff:
#                 line_diff[result_name].set_xdata(lam * 1e6)
#                 line_diff[result_name].set_ydata(I_pred - I_true)

#             # if name == 'ratio':
#             #     # print(len(lam))
#             #     # print(len(results["I"][:, i_T]))
#             #     line_ratio.set_xdata(lam * 1e6)
#             #     line_ratio.set_ydata(results["I"][:, i_T])

#         # title = f"{test_data['T'][i_T] - abs0:.0f}$^\circ$C"
#         # if len(results) == 1:
#         #     result = results[plot_result_names[0]]
#         #     temp = results['T'][i_T] - abs0
#         #     C = results['C'][i_T]
#         #     D = results['D'][i_T]
#         #     title += f" {temp:.0f}"
#         #     for param_name in results['intensity_params']:
#         #         title += f" {param_name} {results[param_name][i_T]:.3f}"
#         # title += " $^\circ$C"
#         # ax.set_title(title)
#         for ax_i in [ax, ax_diff]:
#             if ax_i is None:
#                 continue
#             ax_i.relim()
#             ax_i.autoscale_view()
#             if ylim is not None:
#                 ax_i.set_ylim(ylim)

#         # if manual_fit:
#         #     T_slider.set_val(temp)
#         #     C_slider.set_val(C)
#         #     D_slider.set_val(D)

#         fig.canvas.draw_idle()

#     # def update_man_fit(T=None, C=None, D=None):
#     #     if T is None:
#     #         T = T_slider.val
#     #     if C is None:
#     #         C = C_slider.val
#     #     if D is None:
#     #         D = D_slider.val

#     #     i_T = int(i_slider.val)

#     #     # Only called when a single fit is plotted
#     #     plot_result_name = plot_result_names[0]

#     #     lam = self.test_data['lam']
#     #     results = self.results[plot_result_name]
#     #     func_fit = results[intensity_func_name]
#     #     # params = (results[k][i_T] for k in results["intensity_params"])
#     #     I_man_fit = func_fit(
#     #         lam,
#     #         T + abs0,
#     #         # *params,
#     #         C,
#     #         D,
#     #         self.props["lam_0"],
#     #         self.props["lam_inf"],
#     #     )
#     #     line_man_fit.set_xdata(lam * 1e6)
#     #     line_man_fit.set_ydata(I_man_fit)

#     #     ax.relim()
#     #     ax.autoscale_view()
#     #     if ylim is not None:
#     #         ax.set_ylim(ylim)

#     #     fig.canvas.draw_idle()


#     plot_slider = False
#     if i_T is None:
#         i_T = 0
#         plot_slider = True

#     if plot_diff:
#         fig, axes = plt.subplots(
#             2, 1, figsize=(4, 6), 
#             constrained_layout=(not plot_slider)
#         )
#         ax = axes[0]
#         ax_diff = axes[1]
#     else:
#         fig = plt.figure(figsize=(4, 3), constrained_layout=(not plot_slider))
#         ax = plt.gca()
#         ax_diff = None
#     # fig.suptitle(self.name)

#     (line_true,) = ax.plot([0], [0], alpha=0.5)
#     points = ax.scatter([0], [0], marker=".", alpha=0.5)

#     line_pred = {}
#     for result_name in results.keys():
#         (line_pred[result_name],) = ax.plot([0], [0], label=result_name)


#     # (line_ratio,) = ax.plot([0], [0], label='ratio_extra')


#     ax.set_xlabel("Wavelength ($\mu$m)")
#     ax.set_ylabel("Intensity")
#     ax.legend()

#     if plot_diff:
#         # advance colour cycle to be consistent with other axis
#         ax_diff._get_lines.get_next_color()
#         line_diff = {}
#         for result_name in results.keys():
#             (line_diff[result_name],) = ax_diff.plot(
#                 [0], [0], label=result_name
#             )

#         ax_diff.set_xlabel("Wavelength ($\mu$m)")
#         ax_diff.set_ylabel("Intensity error")
        
#     # update(i_T)

#     rtn = ()

#     if plot_slider:
#         def slider_key(event, slider):
#             if event.key == 'left':
#                 slider.set_val(slider.val - 1)
#             elif event.key == 'right':
#                 slider.set_val(slider.val + 1)

#         fig.subplots_adjust(bottom=0.2)
#         ax_i = fig.add_axes([0.18, 0.02, 0.65, 0.01])
#         i_slider = Slider(
#             ax=ax_i,
#             label="",
#             valmin=0,
#             valmax=len(test_data["I"]) - 1,
#             valstep=1,
#             valinit=i_T,
#         )
#         i_slider.on_changed(update)
#         fig.canvas.mpl_connect('key_press_event', lambda e: slider_key(e, i_slider))
#         # self.fig_widget = i_slider  # keep fig alive
#         rtn = rtn + (i_slider ,)
    
#     # if manual_fit:
#     #     assert len(plot_result_names) == 1
#     #     (line_man_fit, ) = ax.plot([0], [0])

#     #     fig.subplots_adjust(bottom=0.3)
#     #     ax_T = fig.add_axes([0.18, 0.1, 0.65, 0.01])
#     #     ax_C = fig.add_axes([0.18, 0.08, 0.65, 0.01])
#     #     ax_D = fig.add_axes([0.18, 0.06, 0.65, 0.01])
#     #     T_slider = Slider(
#     #         ax=ax_T,label="T",valmin=0,valmax=3000,valinit=300,
#     #     )
#     #     C_slider = Slider(
#     #         ax=ax_C, label="C", valmin=0, valmax=1, valinit=0.5,
#     #     )
#     #     D_slider = Slider(
#     #         ax=ax_D, label="D", valmin=0, valmax=1, valinit=0.5,
#     #     )
#     #     T_slider.on_changed(lambda x: update_man_fit(T=x))
#     #     C_slider.on_changed(lambda x: update_man_fit(C=x))
#     #     D_slider.on_changed(lambda x: update_man_fit(D=x))
#     #     self.fig_widget_2 = T_slider  # keep fig alive
#         # rtn = rtn + (T_slider, )


#     update(i_T)

#     plt.show()

#     return rtn