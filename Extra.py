# Extra
             if temp_y[i-1]-temp_y[i] < 0:
                bi = Akima1DInterpolator([temp_y[i-1], temp_y[i]], [temp_x[i-1], temp_x[i]])
                temp_y_interp = np.linspace(temp_y[i-1], temp_y[i], num=nInterp)
                temp_x_interp = bi(temp_y_interp)
            if temp_y[i-1]-temp_y[i] > 0:
                bi = Akima1DInterpolator([temp_y[i], temp_y[i-1]], [temp_x[i], temp_x[i-1]])
                temp_y_interp = np.linspace(temp_y[i-1], temp_y[i], num=nInterp)
                temp_x_interp_reversed = bi(temp_y_interp[::-1])
                temp_x_interp = temp_x_interp_reversed[::-1]
            if temp_y[i-1]-temp_y[i] == 0: