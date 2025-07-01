      SUBROUTINE DFLUX(FLUX,SOL,JSTEP,JINC,TIME,NOEL,NPT,COORDS,JLTYP,
     $                 TEMP,PRESS,SNAME)
C
      INCLUDE 'ABA_PARAM.INC'
C
      DIMENSION COORDS(3),FLUX(2),TIME(2)
      CHARACTER*80 SNAME
C
      wu=12
      wi=42
      effi=0.85
      v=0.04
      q=wu*wi*effi*25
      d=v*TIME(2)
C
      x=COORDS(1)
      y=COORDS(2)
      z=COORDS(3)
C
      x0=-v
      y0=0 
      z0=5e-4
C
      a1=2*1.8675e-3
      a2=2*3.735e-3
      b=2*1.8675e-3
      c=2*2.763e-3
C
      f1=0.6
      PI=3.1415926
C
      heat1=6.0*sqrt(3.0)*q/(a1*b*c*PI*sqrt(PI))*f1
      heat2=6.0*sqrt(3.0)*q/(a2*b*c*PI*sqrt(PI))*(2.0-f1)
C
      shape1=exp(-3.0*(x-x0-d)**2/(a1)**2-3.0*(y-y0)**2/b**2
     $	-3.0*(z-z0)**2/c**2)
      shape2=exp(-3.0*(x-x0-d)**2/(a2)**2-3.0*(y-y0)**2/b**2
     $	-3.0*(z-z0)**2/c**2)
C
      JLTYP=1
      IF(x .GE.(x0+d)) THEN
        FLUX(1)=heat1*shape1
      ELSE
        FLUX(1)=heat2*shape2
      ENDIF
      RETURN
      END
