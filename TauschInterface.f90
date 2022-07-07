module TauschInterface

    interface
        type(c_ptr) function f_MPI_Comm_f2c(fcomm) &
                        bind(C, name="f_MPI_Comm_f2c")
            use iso_c_binding, only: c_ptr,c_int
            integer(c_int) :: fcomm
        end function
    end interface

    interface
        type(c_ptr) function tausch_new(ccomm, useDuplicateOfCommunicator) &
                        bind(C, name="tausch_new_f")
            use iso_c_binding, only: c_int, c_ptr
            integer(c_int),value :: ccomm
            logical(4),value :: useDuplicateOfCommunicator
        end function
    end interface

    interface
        subroutine tausch_delete(tausch) &
                        bind(C, name="tausch_delete")
            use iso_c_binding, only: c_ptr
            type(c_ptr),value :: tausch
        end subroutine
    end interface

    interface
        subroutine tausch_addSendHaloInfo(tausch, haloIndices, numHaloIndices, typeSize, remoteMpiRank) &
                        bind(C, name="tausch_addSendHaloInfo")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr),value :: tausch
            integer(c_int) :: haloIndices(*)
            integer(c_int),value :: numHaloIndices
            integer(c_int),value :: typeSize
            integer(c_int),value :: remoteMpiRank
        end subroutine tausch_addSendHaloInfo
    end interface

    interface
        subroutine tausch_addRecvHaloInfo(tausch, haloIndices, numHaloIndices, typeSize, remoteMpiRank) &
                        bind(C, name="tausch_addRecvHaloInfo")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr),value :: tausch
            integer(c_int) :: haloIndices(*)
            integer(c_int),value :: numHaloIndices
            integer(c_int),value :: typeSize
            integer(c_int),value :: remoteMpiRank
        end subroutine tausch_addRecvHaloInfo
    end interface

    interface
        subroutine tausch_setSendCommunicationStrategy(tausch, haloId, strategy) &
                        bind(C, name="tausch_setSendCommunicationStrategy")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: strategy
        end subroutine tausch_setSendCommunicationStrategy
    end interface

    interface
        subroutine tausch_setRecvCommunicationStrategy(tausch, haloId, strategy) &
                        bind(C, name="tausch_setRecvCommunicationStrategy")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: strategy
        end subroutine tausch_setRecvCommunicationStrategy
    end interface

    interface
        subroutine tausch_setSendHaloBuffer(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setSendHaloBuffer")
            use iso_c_binding, only: c_ptr, c_int, c_char
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            character(c_char) :: buf(*)
        end subroutine tausch_setSendHaloBuffer
    end interface

    interface
        subroutine tausch_setSendHaloBuffer_double(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setSendHaloBuffer_double")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_double) :: buf(*)
        end subroutine tausch_setSendHaloBuffer_double
    end interface

    interface
        subroutine tausch_setSendHaloBuffer_float(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setSendHaloBuffer_float")
            use iso_c_binding, only: c_ptr, c_int, c_float
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_float) :: buf(*)
        end subroutine tausch_setSendHaloBuffer_float
    end interface

    interface
        subroutine tausch_setSendHaloBuffer_int(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setSendHaloBuffer_int")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            integer(c_int) :: buf(*)
        end subroutine tausch_setSendHaloBuffer_int
    end interface

    interface
        subroutine tausch_setRecvHaloBuffer(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setRecvHaloBuffer")
            use iso_c_binding, only: c_ptr, c_int, c_char
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            character(c_char) :: buf(*)
        end subroutine tausch_setRecvHaloBuffer
    end interface

    interface
        subroutine tausch_setRecvHaloBuffer_double(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setRecvHaloBuffer_double")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_double) :: buf(*)
        end subroutine tausch_setRecvHaloBuffer_double
    end interface

    interface
        subroutine tausch_setRecvHaloBuffer_float(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setRecvHaloBuffer_float")
            use iso_c_binding, only: c_ptr, c_int, c_float
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_float) :: buf(*)
        end subroutine tausch_setRecvHaloBuffer_float
    end interface

    interface
        subroutine tausch_setRecvHaloBuffer_int(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_setRecvHaloBuffer_int")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            integer(c_int) :: buf(*)
        end subroutine tausch_setRecvHaloBuffer_int
    end interface

    interface
        subroutine tausch_packSendBuffer(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_packSendBuffer")
            use iso_c_binding, only: c_ptr, c_int, c_char
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            character(c_char) :: buf(*)
        end subroutine tausch_packSendBuffer
    end interface

    interface
        subroutine tausch_packSendBuffer_double(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_packSendBuffer_double")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_double) :: buf(*)
        end subroutine tausch_packSendBuffer_double
    end interface

    interface
        subroutine tausch_packSendBuffer_float(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_packSendBuffer_float")
            use iso_c_binding, only: c_ptr, c_int, c_double, c_float
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_float) :: buf(*)
        end subroutine tausch_packSendBuffer_float
    end interface

    interface
        subroutine tausch_packSendBuffer_int(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_packSendBuffer_int")
            use iso_c_binding, only: c_ptr, c_int, c_double
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            integer(c_int) :: buf(*)
        end subroutine tausch_packSendBuffer_int
    end interface

    interface
        function tausch_send(tausch, haloId, msgtag, remoteMpiRank, bufferId, blocking, ccomm) &
                        bind(C, name="tausch_send_f")
            use iso_c_binding, only: c_ptr,c_int
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: msgtag
            integer(c_int),value :: remoteMpiRank
            integer(c_int),value :: bufferId
            logical(4),value :: blocking
            integer(c_int),value :: ccomm
        end function
    end interface

    interface
        function tausch_recv(tausch, haloId, msgtag, remoteMpiRank, bufferId, blocking, ccomm) &
                        bind(C, name="tausch_recv_f")
            use iso_c_binding, only: c_ptr,c_int,C_BOOL
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: msgtag
            integer(c_int),value :: remoteMpiRank
            integer(c_int),value :: bufferId
            logical(4),value :: blocking
            integer(c_int),value :: ccomm
        end function
    end interface

    interface
        subroutine tausch_unpackRecvBuffer(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_unpackRecvBuffer")
            use iso_c_binding, only: c_ptr,c_int,c_char
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            character(c_char) :: buf(*)
        end subroutine tausch_unpackRecvBuffer
    end interface

    interface
        subroutine tausch_unpackRecvBuffer_double(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_unpackRecvBuffer_double")
            use iso_c_binding, only: c_ptr,c_int,c_double
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_double) :: buf(*)
        end subroutine tausch_unpackRecvBuffer_double
    end interface

    interface
        subroutine tausch_unpackRecvBuffer_float(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_unpackRecvBuffer_float")
            use iso_c_binding, only: c_ptr,c_int,c_double,c_float
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            real(c_float) :: buf(*)
        end subroutine tausch_unpackRecvBuffer_float
    end interface

    interface
        subroutine tausch_unpackRecvBuffer_int(tausch, haloId, bufferId, buf) &
                        bind(C, name="tausch_unpackRecvBuffer_int")
            use iso_c_binding, only: c_ptr,c_int
            type(c_ptr),value :: tausch
            integer(c_int),value :: haloId
            integer(c_int),value :: bufferId
            integer(c_int) :: buf(*)
        end subroutine tausch_unpackRecvBuffer_int
    end interface

end module TauschInterface
