use core::ops::Range;
use goblin::elf::{
    header::header32::Header, program_header::program_header32::ProgramHeader, program_header::*,
};
use spin::Mutex;

/// Segment is executable.
const PF_X: u32 = 1 << 0;

/// Segment is writable.
const PF_W: u32 = 1 << 1;

/// Native page size.
const PAGE_SIZE: usize = 4096;

/// Read raw bytes into a typed value.
trait ReadRaw {
    /// Perform the read. Unsafe because the caller needs to ensure that any bit pattern
    /// will be a valid `T`.
    unsafe fn read_raw<T: Sized + Copy>(&self) -> Option<T>;
}

impl ReadRaw for [u8] {
    unsafe fn read_raw<T: Sized + Copy>(&self) -> Option<T> {
        if self.len() < core::mem::size_of::<T>() {
            None
        } else {
            Some(*(self.as_ptr() as *const T))
        }
    }
}

/// ELF image metadata.
#[derive(Clone, Debug)]
pub struct ElfMetadata {
    /// The entry virtual address.
    pub entry_address: u32,
}

bitflags! {
    pub struct SegmentProtection: u8 {
        const R = 1 << 0;
        const W = 1 << 1;
        const X = 1 << 2;
    }
}

/// ELF segment metadata.
#[derive(Clone, Debug)]
pub struct SegmentMetadata {
    pub base: u32,
    pub len: u32,
    pub prot: SegmentProtection,
}

pub trait MemoryManager {
    unsafe fn mmap(&mut self, start: usize, len: usize, prot: SegmentProtection) -> bool;
    unsafe fn mprotect(&mut self, start: usize, len: usize, prot: SegmentProtection) -> bool;
}

pub unsafe fn load<M: MemoryManager>(
    image: &[u8],
    va_space: Range<usize>,
    mm: &mut M,
) -> Option<ElfMetadata> {
    let header: Header = match image.read_raw() {
        Some(x) => x,
        None => return None,
    };
    if header.e_phoff >= image.len() as u32 {
        return None;
    }

    let mut segments = &image[header.e_phoff as usize..];
    for _ in 0..header.e_phnum {
        let ph: ProgramHeader = segments.read_raw()?;
        segments = &segments[core::mem::size_of::<ProgramHeader>()..];
        if ph.p_type != PT_LOAD {
            continue;
        }
        let mut padding_before: usize = 0;
        let start = ph.p_vaddr as usize;
        if start % PAGE_SIZE != 0 {
            padding_before = start % PAGE_SIZE;
        }
        if ph.p_filesz > ph.p_memsz {
            return None;
        }
        let mut mem_end = start.checked_add(ph.p_memsz as usize)?;
        let file_end = start.checked_add(ph.p_filesz as usize)?;
        if file_end - start > image.len() {
            return None;
        }

        if mem_end % PAGE_SIZE != 0 {
            mem_end = (mem_end / PAGE_SIZE + 1) * PAGE_SIZE;
        }

        let mut prot = SegmentProtection::R;
        if ph.p_flags & PF_W != 0 {
            prot |= SegmentProtection::W;
        }
        /*if ph.p_flags & PF_X != 0 {
            prot |= SegmentProtection::X;
        }*/

        let va_begin = (start - padding_before).checked_add(va_space.start)?;
        let va_end = va_begin.checked_add(mem_end - (start - padding_before))?;
        if va_end > va_space.end {
            return None;
        }

        if !mm.mmap(
            va_begin,
            va_end - va_begin,
            SegmentProtection::R | SegmentProtection::W,
        ) {
            return None;
        }

        let slice = core::slice::from_raw_parts_mut(va_begin as *mut u8, va_end - va_begin);

        slice[padding_before..padding_before + (ph.p_filesz as usize)].copy_from_slice(
            &image[(ph.p_offset as usize)..((ph.p_offset as usize) + (ph.p_filesz as usize))],
        );
        if !mm.mprotect(va_begin, va_end - va_begin, prot) {
            return None;
        }
    }
    Some(ElfMetadata {
        entry_address: header.e_entry,
    })
}
