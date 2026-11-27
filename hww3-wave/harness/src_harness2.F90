!> Source-term harness of the ice and reflection terms of the IRI
!> configuration (W3SIS2, W3SIC2 and W3SREF; switches IS2, IC2 and REF1),
!> companion of src_harness.F90: compiled against a WW3 library built with
!> those switches (the IRI build), it calls the terms node by node on the
!> spectra of a WW3 restart of a structured grid and the wind of a
!> ww3_prnc forcing file, with the ice thickness, floe sizes, iceberg
!> field and reflection arrays that W3WAVE passes to W3SRCE, and writes
!> their inputs and outputs, so that hww3_wave::ice and hww3_wave::ref1
!> can be checked node by node.
!>
!> usage (in a directory holding the mod_def.ww3 of the grid):
!>   ww3_srcharness2 RESTART WIND OUT STEP
!> RESTART  a FULL restart of the grid (spectra, level, ice, u*, ASF)
!> WIND     wind.ww3 of ww3_prnc: the first record at or after the restart
!>          time is used, as the absolute 10 m wind
!> OUT      output file (stream, native little-endian), layout below
!> STEP     every STEP-th sea point (MAPSTA = 1) with ice, and every
!>          STEP-th one with a reflection coefficient, are taken
!>
!> As W3WAVE: ICEH = ICEHINIT (no thickness forcing), ICEF = ICEDMAX =
!> 1000 (initial values, no floe-size forcing), BERG = 0, REFLEC =
!> REFLC(:, ISEA) with REFLEC(4) = BERG REFLC(4, ISEA), REFLED =
!> REFLD(:, ISEA), DELX, DELY, DELA from the grid spacing (SX CLATS / FACX,
!> SY / FACX on regular grids, HPFAC / FACX, HQFAC / FACX on curvilinear
!> ones), TRNX(IY, IX), TRNY(IY, IX); WN_R = WN, CG_ICE = CG (no ice
!> dispersion); the input field ICEI(IX, IY) read by W3SIC2 holds the
!> ice concentration of the node.
!>
!> Output: header "HWW3SRC2", NK, NTH, NSPEC, NNODE, NREC (int32), restart
!> time (2 int32), wind time (2 int32), DTG (real32); then NNODE records of
!> ISEA (int32) followed by NREC real32 values:
!>   inputs  IX IY DW DEPTH ICE ICEH ICEF ICEDMAX BERG U10 U10DIR UST0
!>           USTDIR0 ASF FPIS0 D50 PSIC DELX DELY DELA TRNX TRNY
!>           REFLEC(4) REFLED(6) CLATS WN(NK) CG(NK) SPEC0(NSPEC)
!>   spr4    EMEAN FMEAN (W3SPR4 of SPEC0 with the restart u*)
!>   sis2    S(NSPEC) D(NSPEC) DISSIP(NSPEC) R(NK) ICEF (after the call)
!>   sic2    S(NSPEC) D(NSPEC) (with the R of W3SIS2)
!>   sref    S(NSPEC) (W3SREF on SPEC0 with DT = DTG)
!>   srce    SPEC(NSPEC) DTDYN ICEF PHICE TAUICE(2) USTAR (a full W3SRCE,
!>           DTG = 600 s, from SPEC0)
!>   srce0   SPEC(NSPEC) DTDYN (the same W3SRCE with ICE = 0 and no
!>           reflection)
PROGRAM HWW3_SRC_HARNESS2
  USE CONSTANTS, ONLY: DWAT, DAIR, TPI, SRCE_DIRECT, DERA, RADIUS
  USE W3GDATMD, ONLY: W3NMOD, W3SETG, NK, NTH, NSPEC, SIG, NSEA, ZB, CLATS, &
       DMIN, MAPSF, MAPSTA, GTYPE, RLGTYPE, CLGTYPE, SX, SY, HPFAC, HQFAC, &
       TRNX, TRNY, REFLC, REFLD, IICEHINIT, SED_D50, SED_PSIC, FLAGLL
  USE W3WDATMD, ONLY: W3NDAT, W3SETW
  USE W3ADATMD, ONLY: W3NAUX, W3SETA
  USE W3ODATMD, ONLY: W3NOUT, W3SETO, NAPROC, IAPROC
  USE W3IDATMD, ONLY: W3NINP, W3SETI, W3DIMI, INFLAGS1, INFLAGS2, ICEI
  USE W3IOGRMD, ONLY: W3IOGR
  USE W3SRCEMD, ONLY: W3SRCE
  USE W3SRC4MD, ONLY: W3SPR4
  USE W3SIS2MD, ONLY: W3SIS2
  USE W3SIC2MD, ONLY: W3SIC2
  USE W3REF1MD, ONLY: W3SREF
  USE W3DISPMD, ONLY: WAVNU3
  USE MPI_F08, ONLY: MPI_INIT, MPI_FINALIZE
  IMPLICIT NONE
  CHARACTER(LEN=1024) :: FRST, FWND, FOUT, ARG
  INTEGER :: NSTEP, NARG, I, ISEA, IK, ITH, IP, NSIZE, NPART, LRECL, IERR, IOS
  INTEGER :: NNODE, NREC, POS, IX, IY, N0, N1, NICE, NREF
  INTEGER :: RTIME(2), WTIME(2)
  INTEGER, ALLOCATABLE :: NODES(:), MAPTMP(:), ITMP(:)
  LOGICAL, ALLOCATABLE :: TAKE(:)
  REAL, ALLOCATABLE :: WLV(:), ICEA(:), UST(:), USTD(:), ASF(:), FPIS(:), UX(:), UY(:), TMP(:)
  REAL, ALLOCATABLE :: REC(:)
  REAL :: DW, DEPTH, DEPTHBAT, WNBAT, CGBAT, U10, U10D, ICE, COEF, FPI, D50, PSIC
  REAL, ALLOCATABLE :: WN1(:), CG1(:), WN2(:), SPEC0(:), SPEC(:), ALPHA(:), R(:), ALIU(:)
  REAL, ALLOCATABLE :: VS(:), VD(:), DISS(:), SPECOLD(:), VSIO(:), VDIO(:)
  LOGICAL, ALLOCATABLE :: LLWS(:)
  REAL :: EMEAN, FMEAN, FMEAN1, WNMEAN, AMAX, USTAR, USTDIR, TAUWX, TAUWY, CD, Z0, CHARN
  REAL :: FMEANWS, DLWMEAN, DRAT, AS
  REAL :: WHITECAP(4), TAUBBL(2), BEDFORM(3), TAUICE(2)
  LOGICAL :: SHAVEIO
  REAL :: DTDYN, FCUT, TAUOX, TAUOY, TAUWIX, TAUWIY, TAUWNX, TAUWNY, PHIAW, TWS, PHIOC
  REAL :: PHIBBL, PHICE, TAUOCX, TAUOCY, REFLEC(4), BERG, ICEH, ICEF, ICEDMAX
  REAL :: DELX, DELY, DELA, FACX, TRX, TRY
  INTEGER :: REFLED(6)
  REAL, PARAMETER :: DTG = 600.

  CALL MPI_INIT(IERR)
  NARG = COMMAND_ARGUMENT_COUNT()
  IF (NARG .LT. 4) STOP 'usage: ww3_srcharness2 RESTART WIND OUT STEP'
  CALL GET_COMMAND_ARGUMENT(1, FRST)
  CALL GET_COMMAND_ARGUMENT(2, FWND)
  CALL GET_COMMAND_ARGUMENT(3, FOUT)
  CALL GET_COMMAND_ARGUMENT(4, ARG)
  READ (ARG, *) NSTEP

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
  INFLAGS2(4) = .TRUE.          ! ice concentration forcing, as in the runs
  ! the input field of ice concentrations (W3SIC2 reads ICEI(IX, IY))
  INFLAGS1(4) = .TRUE.
  CALL W3DIMI(1, 6, 6, (/ .FALSE., .FALSE., .FALSE., .FALSE. /))
  CALL W3SETI(1, 6, 6)
  ICEI = 0.
  WRITE (*, '(A,I8,A,2I4,A,I2)') ' mod_def read: NSEA', NSEA, ', NK NTH', NK, NTH, ', GTYPE', GTYPE
  FACX = 1.
  IF (FLAGLL) FACX = 1. / (DERA * RADIUS)

  ! restart: spectra, then level, ice, MAPSTA, u*, u* direction, ASF, FPIS
  LRECL = MAX(4 * NSPEC, 4 * (6 + (25 / 4) + (9 / 4) + (29 / 4) + (3 / 4)))
  NSIZE = LRECL / 4
  NPART = 1 + (NSEA - 1) / NSIZE
  ALLOCATE (WLV(NSEA), ICEA(NSEA), MAPTMP(NSEA), UST(NSEA), USTD(NSEA), ASF(NSEA), FPIS(NSEA))
  ALLOCATE (TMP(NSIZE), ITMP(NSIZE))
  OPEN (20, FILE=TRIM(FRST), FORM='UNFORMATTED', ACCESS='DIRECT', RECL=LRECL, &
       CONVERT='BIG_ENDIAN', STATUS='OLD', ACTION='READ')
  READ (20, REC=2) RTIME
  CALL READ_PARTS(NSEA + 4, WLV)
  CALL READ_PARTS(NSEA + 4 + NPART, ICEA)
  DO IP = 1, NPART
    N0 = 1 + (IP - 1) * NSIZE
    N1 = MIN(NSEA, IP * NSIZE)
    READ (20, REC=NSEA + 4 + 2 * NPART + IP - 1) ITMP(1:N1 - N0 + 1)
    MAPTMP(N0:N1) = ITMP(1:N1 - N0 + 1)
  END DO
  CALL READ_PARTS(NSEA + 4 + 3 * NPART, UST)
  CALL READ_PARTS(NSEA + 4 + 4 * NPART, USTD)
  CALL READ_PARTS(NSEA + 4 + 5 * NPART, ASF)
  CALL READ_PARTS(NSEA + 4 + 6 * NPART, FPIS)
  WRITE (*, '(A,2I9)') ' restart time', RTIME

  ! wind: first record at or after the restart time, on the NX x NY grid
  OPEN (21, FILE=TRIM(FWND), FORM='UNFORMATTED', ACCESS='SEQUENTIAL', &
       CONVERT='BIG_ENDIAN', STATUS='OLD', ACTION='READ')
  READ (21)
  ALLOCATE (UX(SIZE(MAPSTA)), UY(SIZE(MAPSTA)))
  DO
    READ (21, IOSTAT=IOS) WTIME
    IF (IOS .NE. 0) STOP 'wind file ends before the restart time'
    READ (21) UX
    READ (21) UY
    IF (WTIME(1) .GT. RTIME(1) .OR. (WTIME(1) .EQ. RTIME(1) .AND. WTIME(2) .GE. RTIME(2))) EXIT
  END DO
  WRITE (*, '(A,2I9)') ' wind time', WTIME

  ! nodes: every NSTEP-th sea point with ice, every NSTEP-th with reflection
  ALLOCATE (TAKE(NSEA))
  TAKE = .FALSE.
  NICE = 0
  NREF = 0
  DO ISEA = 1, NSEA
    IF (MOD(MAPTMP(ISEA) + 2, 8) - 2 .NE. 1) CYCLE
    IF (ICEA(ISEA) .GT. 0.) THEN
      NICE = NICE + 1
      IF (MOD(NICE - 1, NSTEP) .EQ. 0) TAKE(ISEA) = .TRUE.
    END IF
    IF (REFLC(1, ISEA) .GT. 0.) THEN
      NREF = NREF + 1
      IF (MOD(NREF - 1, NSTEP) .EQ. 0) TAKE(ISEA) = .TRUE.
    END IF
  END DO
  NNODE = COUNT(TAKE)
  WRITE (*, '(A,I8,A,I8,A,I8,A)') ' sea points with ice', NICE, ', with reflection', NREF, ', taken', NNODE
  ALLOCATE (NODES(NNODE))
  NNODE = 0
  DO ISEA = 1, NSEA
    IF (TAKE(ISEA)) THEN
      NNODE = NNODE + 1
      NODES(NNODE) = ISEA
    END IF
  END DO

  NREC = (33 + 2 * NK + NSPEC) + 2 + (3 * NSPEC + NK + 1) + 2 * NSPEC + NSPEC &
       + (NSPEC + 6) + (NSPEC + 1)
  ALLOCATE (REC(NREC))
  ALLOCATE (WN1(NK), CG1(NK), WN2(NSPEC), SPEC0(NSPEC), SPEC(NSPEC), ALPHA(NK), R(NK), ALIU(NK))
  ALLOCATE (VS(NSPEC), VD(NSPEC), DISS(NSPEC), LLWS(NSPEC), SPECOLD(NSPEC), VSIO(NSPEC), VDIO(NSPEC))
  OPEN (30, FILE=TRIM(FOUT), ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE')
  WRITE (30) 'HWW3SRC2', NK, NTH, NSPEC, NNODE, NREC, RTIME, WTIME, DTG
  DRAT = DAIR / DWAT
  AS = 0.
  ALIU = 0.
  DO I = 1, NNODE
    ISEA = NODES(I)
    IX = MAPSF(ISEA, 1)
    IY = MAPSF(ISEA, 2)
    POS = 0
    DW = MAX(0., WLV(ISEA) - ZB(ISEA))
    DEPTHBAT = MAX(DMIN, -ZB(ISEA))
    CALL WAVNU3(SIG(1), DEPTHBAT, WNBAT, CGBAT)
    IF (WNBAT * DEPTHBAT .GE. 4.) THEN
      DEPTH = DEPTHBAT
    ELSE
      DEPTH = MAX(DMIN, DW)
    END IF
    DO IK = 1, NK
      CALL WAVNU3(SIG(IK), DEPTH, WN1(IK), CG1(IK))
      DO ITH = 1, NTH
        WN2(ITH + (IK - 1) * NTH) = WN1(IK)
      END DO
    END DO
    DEPTH = MAX(DMIN, DW)
    READ (20, REC=ISEA + 2) SPEC0
    U10 = SQRT(UX(IX + (IY - 1) * SIZE(MAPSTA, 2))**2 + UY(IX + (IY - 1) * SIZE(MAPSTA, 2))**2)
    U10D = MOD(TPI + ATAN2(UY(IX + (IY - 1) * SIZE(MAPSTA, 2)), UX(IX + (IY - 1) * SIZE(MAPSTA, 2))), TPI)
    ICE = ICEA(ISEA)
    ICEI(IX, IY) = ICE
    COEF = ASF(ISEA)
    FPI = FPIS(ISEA)
    D50 = SED_D50(ISEA)
    PSIC = SED_PSIC(ISEA)
    ICEH = IICEHINIT
    ICEF = 1000.
    ICEDMAX = 1000.
    BERG = 0.
    DELA = 1.
    DELX = 1.
    DELY = 1.
    IF (GTYPE .EQ. RLGTYPE) THEN
      DELX = SX * CLATS(ISEA) / FACX
      DELY = SY / FACX
      DELA = DELX * DELY
    END IF
    IF (GTYPE .EQ. CLGTYPE) THEN
      DELX = HPFAC(IY, IX) / FACX
      DELY = HQFAC(IY, IX) / FACX
      DELA = DELX * DELY
    END IF
    REFLEC = REFLC(:, ISEA)
    REFLEC(4) = BERG * REFLEC(4)
    REFLED = REFLD(:, ISEA)
    TRX = TRNX(IY, IX)
    TRY = TRNY(IY, IX)
    CALL PUT(REAL(IX)); CALL PUT(REAL(IY)); CALL PUT(DW); CALL PUT(DEPTH); CALL PUT(ICE)
    CALL PUT(ICEH); CALL PUT(ICEF); CALL PUT(ICEDMAX); CALL PUT(BERG); CALL PUT(U10)
    CALL PUT(U10D); CALL PUT(UST(ISEA)); CALL PUT(USTD(ISEA)); CALL PUT(COEF); CALL PUT(FPI)
    CALL PUT(D50); CALL PUT(PSIC); CALL PUT(DELX); CALL PUT(DELY); CALL PUT(DELA)
    CALL PUT(TRX); CALL PUT(TRY); CALL PUTV(REFLEC); CALL PUTV(REAL(REFLED))
    CALL PUT(CLATS(ISEA))
    CALL PUTV(WN1); CALL PUTV(CG1); CALL PUTV(SPEC0)

    ! mean parameters of the spectrum (for W3SREF)
    SPEC = SPEC0
    USTAR = UST(ISEA)
    USTDIR = USTD(ISEA)
    TAUWX = 0.
    TAUWY = 0.
    LLWS = .TRUE.
    CALL W3SPR4(SPEC, CG1, WN1, EMEAN, FMEAN, FMEAN1, WNMEAN, AMAX, U10, U10D, &
         USTAR, USTDIR, TAUWX, TAUWY, CD, Z0, CHARN, LLWS, FMEANWS, DLWMEAN)
    CALL PUT(EMEAN); CALL PUT(FMEAN)

    ! W3SIS2 then W3SIC2 (R of W3SIS2), as after the integration loop
    R = 1.
    CALL W3SIS2(SPEC, DEPTH, ICE, ICEH, ICEF, ICEDMAX, IX, IY, VS, VD, DISS, &
         WN1, CG1, WN1, CG1, R)
    CALL PUTV(VS); CALL PUTV(VD); CALL PUTV(DISS); CALL PUTV(R); CALL PUT(ICEF)
    ICEF = 1000.
    CALL W3SIC2(SPEC, DEPTH, ICEH, ICEF, CG1, WN1, IX, IY, VS, VD, WN1, CG1, ALIU, R)
    CALL PUTV(VS); CALL PUTV(VD)

    ! W3SREF on the same spectrum
    SPEC = SPEC0
    VS = 0.
    IF (REFLEC(1) .GT. 0. .OR. REFLEC(2) .GT. 0. .OR. (REFLEC(4) .GT. 0. .AND. BERG .GT. 0.)) THEN
      CALL W3SREF(SPEC, CG1, WN1, EMEAN, FMEAN, DEPTH, 0., 0., REFLEC, REFLED, &
           TRX, TRY, BERG, DTG, IX, IY, ISEA, VS)
    END IF
    CALL PUTV(VS)

    ! a full W3SRCE, with ice and reflection, then without them
    CALL FULL_SRCE(ICE, REFLEC)
    CALL PUTV(SPEC); CALL PUT(DTDYN); CALL PUT(ICEF); CALL PUT(PHICE); CALL PUTV(TAUICE)
    CALL PUT(USTAR)
    REFLEC = 0.
    CALL FULL_SRCE(0., REFLEC)
    CALL PUTV(SPEC); CALL PUT(DTDYN)
    IF (POS .NE. NREC) THEN
      WRITE (*, *) 'record length', POS, 'expected', NREC
      STOP 'record layout error'
    END IF
    WRITE (30) ISEA, REC
  END DO
  CLOSE (30)
  WRITE (*, '(A,I8,A,I6,A)') ' wrote', NNODE, ' nodes of', NREC, ' values'
  CALL MPI_FINALIZE(IERR)

CONTAINS

  !> A full W3SRCE from SPEC0, the restart u* and the inputs of the node,
  !> with ice concentration CI and reflection coefficients RC.
  SUBROUTINE FULL_SRCE(CI, RC)
    REAL, INTENT(IN) :: CI, RC(4)
    REAL :: RL(4)
    RL = RC
    SPEC = SPEC0
    SPECOLD = SPEC0
    USTAR = UST(ISEA)
    USTDIR = USTD(ISEA)
    TAUWX = 0.
    TAUWY = 0.
    ALPHA = 0.
    WHITECAP = 0.
    TAUBBL = 0.
    TAUICE = 0.
    BEDFORM = 0.
    PHICE = 0.
    ICEH = IICEHINIT
    ICEF = 1000.
    ICEDMAX = 1000.
    CALL W3SRCE(SRCE_DIRECT, 1, ISEA, ISEA, IX, IY, 1, SPECOLD, SPEC, VSIO, VDIO, &
         SHAVEIO, ALPHA, WN1, CG1, CLATS(ISEA), DW, U10, U10D, AS, USTAR, USTDIR, &
         0., 0., CI, ICEH, ICEF, ICEDMAX, RL, REFLED, DELX, DELY, DELA, TRX, TRY, BERG, &
         FPI, DTDYN, FCUT, DTG, TAUWX, TAUWY, TAUOX, TAUOY, TAUWIX, TAUWIY, TAUWNX, &
         TAUWNY, PHIAW, CHARN, TWS, PHIOC, WHITECAP, D50, PSIC, BEDFORM, PHIBBL, &
         TAUBBL, TAUICE, PHICE, TAUOCX, TAUOCY, WNMEAN, DAIR, COEF)
  END SUBROUTINE FULL_SRCE

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

  SUBROUTINE PUT(X)
    REAL, INTENT(IN) :: X
    POS = POS + 1
    IF (POS .LE. NREC) REC(POS) = X
  END SUBROUTINE PUT

  SUBROUTINE PUTV(X)
    REAL, INTENT(IN) :: X(:)
    INTEGER :: J
    DO J = 1, SIZE(X)
      CALL PUT(X(J))
    END DO
  END SUBROUTINE PUTV

END PROGRAM HWW3_SRC_HARNESS2
