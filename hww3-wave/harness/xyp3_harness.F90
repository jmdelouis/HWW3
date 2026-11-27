!> Harness of the geographic propagation of the structured grids (W3XYP3
!> with the UQ scheme and the GSE averaging; switches PR3 and UQ): compiled
!> against a WW3 library built with those switches (the IRI build), it does
!> what W3INIT and W3WAVE do before the first propagation of a run started
!> from a restart (the status map, levels and ice of the restart, the group
!> velocities of W3INIT, W3MAP3, W3UTRN and W3MAPT), then calls W3XYP3 on a
!> few spectral components for the first two global steps, and writes the
!> maps, the transparencies and the fields before and after, so that
!> hww3_wave::pro3 can be checked point by point (hww3-pr3 --harness).
!>
!> usage (in a directory holding the mod_def.ww3 of the grid):
!>   ww3_xyp3harness RESTART OUT DTG
!> RESTART  a FULL restart of the grid
!> OUT      output file (stream, native little-endian), layout below
!> DTG      the global time step (s): TIME = restart time + DTG for the first
!>          call, + 2 DTG for the second, as after TICK21 in W3WAVE
!>
!> Output: "HWW3XYP3", NX, NY, NSEA, NK, NTH, NISP (int32), restart time (2
!> int32), DTG (real32), NCENT, NMX0, NMX1, NMX2, NMY0, NMY1, NMY2, NACT
!> (int32); MAPSTA(NY*NX) (int32, IXY order); then per sea point (ISEA
!> order) the real32 arrays ICE, WLV, ATRNX(-1), ATRNX(1), ATRNY(-1),
!> ATRNY(1), MAPTRN (0 or 1); then NISP records: ISP (int32), CG(IK, 1:NSEA),
!> the field at the sea points before, after the first W3XYP3 and after the
!> second (real32, NSEA each).
PROGRAM HWW3_XYP3_HARNESS
  USE W3GDATMD, ONLY: W3NMOD, W3SETG, NK, NTH, NSPEC, SIG, NSEA, NX, NY, ZB, &
       DMIN, MAPSF, MAPSTA, MAPST2, MAPFS, TRNX, TRNY
  USE W3WDATMD, ONLY: W3NDAT, W3SETW, W3DIMW, TIME, ICE, WLV, BERG
  USE W3ADATMD, ONLY: W3NAUX, W3SETA, W3DIMA, CG, WN, DW, ATRNX, ATRNY, MAPTRN, &
       NCENT, NMX0, NMX1, NMX2, NMY0, NMY1, NMY2, NACT, MAPTH2
  USE W3ODATMD, ONLY: W3NOUT, W3SETO, NAPROC, IAPROC, FLBPI
  USE W3IDATMD, ONLY: W3NINP, W3SETI, FLCUR
  USE W3IOGRMD, ONLY: W3IOGR
  USE W3DISPMD, ONLY: WAVNU1
  USE W3TIMEMD, ONLY: TICK21
  USE W3UPDTMD, ONLY: W3UTRN
  USE W3PRO3MD, ONLY: W3MAP3, W3MAPT, W3XYP3
  USE MPI_F08, ONLY: MPI_INIT, MPI_FINALIZE
  IMPLICIT NONE
  CHARACTER(LEN=1024) :: FRST, FOUT, ARG
  INTEGER, PARAMETER :: NIK = 3, NITH = 5
  INTEGER, PARAMETER :: IKS(NIK) = (/ 1, 12, 28 /), ITHS(NITH) = (/ 1, 5, 10, 19, 28 /)
  INTEGER :: ISEA, IK, ITH, ISP, IP, NSIZE, NPART, NPRTX, LRECL, IERR, N0, N1, IY, I, J, IXY
  INTEGER :: RTIME(2), NREC
  INTEGER, ALLOCATABLE :: MAPTMP(:,:), ITMP(:)
  REAL, ALLOCATABLE :: TMP(:), SPEC(:), FLD(:,:), FIELD(:), OUTV(:), WLVA(:), ICEA(:)
  REAL :: DTG, DEPTH

  CALL MPI_INIT(IERR)
  IF (COMMAND_ARGUMENT_COUNT() .LT. 3) STOP 'usage: ww3_xyp3harness RESTART OUT DTG'
  CALL GET_COMMAND_ARGUMENT(1, FRST)
  CALL GET_COMMAND_ARGUMENT(2, FOUT)
  CALL GET_COMMAND_ARGUMENT(3, ARG)
  READ (ARG, *) DTG

  CALL W3NMOD(1, 6, 6)
  CALL W3NDAT(6, 6)
  CALL W3NAUX(6, 6)
  CALL W3NOUT(6, 6)
  CALL W3NINP(6, 6)
  CALL W3SETG(1, 6, 6)
  CALL W3SETW(1, 6, 6)
  CALL W3SETA(1, 6, 6)
  CALL W3SETO(1, 6, 6)
  CALL W3SETI(1, 6, 6)
  NAPROC = 1
  IAPROC = 1
  CALL W3IOGR('READ', 11, 1)
  CALL W3DIMW(1, 6, 6, .TRUE.)
  CALL W3SETW(1, 6, 6)
  CALL W3DIMA(1, 6, 6)
  CALL W3SETA(1, 6, 6)
  FLCUR = .FALSE.
  FLBPI = .FALSE.
  BERG = 0.
  MAPTH2 = 0
  WRITE (*, '(A,I9,A,2I5,A,2I4)') ' mod_def read: NSEA', NSEA, ', NX NY', NX, NY, ', NK NTH', NK, NTH

  ! restart (W3IORS, READ, FULL): time, levels, ice and the status map of
  ! the whole grid, row by row
  LRECL = MAX(4 * NSPEC, 4 * (6 + (25 / 4) + (9 / 4) + (29 / 4) + (3 / 4)))
  NSIZE = LRECL / 4
  NPART = 1 + (NSEA - 1) / NSIZE
  NPRTX = 1 + (NX - 1) / NSIZE
  ALLOCATE (TMP(NSIZE), ITMP(NSIZE), MAPTMP(NY, NX), WLVA(NSEA), ICEA(NSEA))
  OPEN (20, FILE=TRIM(FRST), FORM='UNFORMATTED', ACCESS='DIRECT', RECL=LRECL, &
       CONVERT='BIG_ENDIAN', STATUS='OLD', ACTION='READ')
  READ (20, REC=2) RTIME
  CALL READ_PARTS(NSEA + 4, WLVA)
  CALL READ_PARTS(NSEA + 4 + NPART, ICEA)
  WLV(1:NSEA) = WLVA
  ICE(1:NSEA) = ICEA
  NREC = NSEA + 3 + 2 * NPART
  DO IY = 1, NY
    DO IP = 1, NPRTX
      NREC = NREC + 1
      N0 = 1 + (IP - 1) * NSIZE
      N1 = MIN(NX, IP * NSIZE)
      READ (20, REC=NREC) ITMP(1:N1 - N0 + 1)
      MAPTMP(IY, N0:N1) = ITMP(1:N1 - N0 + 1)
    END DO
  END DO
  MAPSTA = MOD(MAPTMP + 2, 8) - 2
  MAPST2 = (MAPTMP - MAPSTA) / 8
  WRITE (*, '(A,2I9,A,I8,A,I9)') ' restart time', RTIME, ', points with ice', COUNT(ICE(1:NSEA) .GT. 0.), &
       ', MAPSTA /= 0', COUNT(MAPSTA .NE. 0)

  ! W3INIT 5.b: depths, wavenumbers and group velocities
  DO ISEA = 1, NSEA
    DW(ISEA) = MAX(0., WLV(ISEA) - ZB(ISEA))
    DEPTH = MAX(DMIN, DW(ISEA))
    DO IK = 0, NK + 1
      CALL WAVNU1(SIG(IK), DEPTH, WN(IK, ISEA), CG(IK, ISEA))
    END DO
  END DO

  ! W3WAVE 3.5 (FLMAP)
  CALL W3MAP3
  CALL W3UTRN(TRNX, TRNY)
  CALL W3MAPT

  OPEN (30, FILE=TRIM(FOUT), ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE')
  WRITE (30) 'HWW3XYP3', NX, NY, NSEA, NK, NTH, NIK * NITH, RTIME, DTG, &
       NCENT, NMX0, NMX1, NMX2, NMY0, NMY1, NMY2, NACT
  WRITE (30) ((MAPSTA(IY, I), IY = 1, NY), I = 1, NX)
  ALLOCATE (OUTV(NSEA))
  WRITE (30) ICE(1:NSEA)
  WRITE (30) WLV(1:NSEA)
  DO J = -1, 1, 2
    DO ISEA = 1, NSEA
      OUTV(ISEA) = ATRNX(MAPSF(ISEA, 3), J)
    END DO
    WRITE (30) OUTV
  END DO
  DO J = -1, 1, 2
    DO ISEA = 1, NSEA
      OUTV(ISEA) = ATRNY(MAPSF(ISEA, 3), J)
    END DO
    WRITE (30) OUTV
  END DO
  DO ISEA = 1, NSEA
    OUTV(ISEA) = MERGE(1., 0., MAPTRN(MAPSF(ISEA, 3)))
  END DO
  WRITE (30) OUTV

  ! the chosen components of the restart spectra
  ALLOCATE (SPEC(NSPEC), FLD(NSEA, NIK * NITH), FIELD(1 - NY:NY * (NX + 2)))
  DO ISEA = 1, NSEA
    READ (20, REC=ISEA + 2) SPEC
    J = 0
    DO I = 1, NIK
      DO IP = 1, NITH
        J = J + 1
        FLD(ISEA, J) = SPEC(ITHS(IP) + (IKS(I) - 1) * NTH)
      END DO
    END DO
  END DO
  CLOSE (20)

  ! W3GATH, W3XYP3 and W3SCAT for two global steps
  J = 0
  DO I = 1, NIK
    DO IP = 1, NITH
      J = J + 1
      IK = IKS(I)
      ITH = ITHS(IP)
      ISP = ITH + (IK - 1) * NTH
      WRITE (30) ISP
      WRITE (30) CG(IK, 1:NSEA)
      WRITE (30) FLD(:, J)
      TIME = RTIME
      DO N0 = 1, 2
        CALL TICK21(TIME, DTG)
        FIELD = 0.
        DO ISEA = 1, NSEA
          FIELD(MAPSF(ISEA, 3)) = FLD(ISEA, J)
        END DO
        CALL W3XYP3(ISP, DTG, MAPSTA, MAPFS, FIELD, 0., 0.)
        DO ISEA = 1, NSEA
          IXY = MAPSF(ISEA, 3)
          IF (MAPSTA(MAPSF(ISEA, 2), MAPSF(ISEA, 1)) .NE. 0) FLD(ISEA, J) = FIELD(IXY)
        END DO
        WRITE (30) FLD(:, J)
      END DO
      WRITE (*, '(A,I6,A,I3,A,I3)') ' ISP', ISP, ' IK', IK, ' ITH', ITH
    END DO
  END DO
  CLOSE (30)
  CALL MPI_FINALIZE(IERR)

CONTAINS

  SUBROUTINE READ_PARTS(R0, A)
    INTEGER, INTENT(IN) :: R0
    REAL, INTENT(OUT) :: A(:)
    INTEGER :: JP, M0, M1
    DO JP = 1, NPART
      M0 = 1 + (JP - 1) * NSIZE
      M1 = MIN(NSEA, JP * NSIZE)
      READ (20, REC=R0 + JP - 1) TMP(1:M1 - M0 + 1)
      A(M0:M1) = TMP(1:M1 - M0 + 1)
    END DO
  END SUBROUTINE READ_PARTS

END PROGRAM HWW3_XYP3_HARNESS
