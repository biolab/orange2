#include "blas.h"

int dcopy_(int *n, double *dx, int *incx, double *dy, int *incy)
{
	int i, ix, iy, m, mp1;
	int nn = *n, iincx = *incx, iincy = *incy;
	if (*n <= 0)
		return;
	if (iincx == 1 && iincy == 1)
	{
		m = nn % 7;
		if (m != 0){
			for (i = 0; i < m; i++)
				dy[i] = dx[i];
			if (nn < 7)
				return 0;
		}

		for (i = m; i < nn; i+=7)
		{
			dy[i] = dx[i];
			dy[i + 1] = dx[i + 1];
			dy[i + 2] = dx[i + 2];
			dy[i + 3] = dx[i + 3];
			dy[i + 4] = dx[i + 4];
			dy[i + 5] = dx[i + 5];
			dy[i + 6] = dx[i + 6];
		}
	}
	else
	{
		ix = (iincx < 0)? -nn*iincx : 0;
		iy = (iincy < 0)? -nn*iincy : 0;
		for (i = 0; i < nn; i++, ix += iincx, iy += iincy)
			dy[iy] = dx[ix];
	}
	return 0;
}
