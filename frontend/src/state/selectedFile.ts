import { FileTextIcon, FileIcon, HardHat, Pointer, TriangleAlert } from "lucide-react";
import { create } from "zustand";

export const DOCUMENTS = [
  {
    title: "Certificates of Occupancy",
    url: "data/Certificates of Occupancy",
    icon: FileIcon,
    isActive: true,
    items: [
      {
        title: "161 W. 56th Street",
        url: "data/Certificates of Occupancy/161 W. 56th Street",
        items: [
          {
            title: "M000093215.PDF",
            url: "data/Certificates of Occupancy/161 W. 56th Street/M000093215.PDF",
            icon: FileTextIcon,
          },
          {
            title: "M00093381B.PDF",
            url: "data/Certificates of Occupancy/161 W. 56th Street/M00093381B.PDF",
            icon: FileTextIcon,
          },
          {
            title: "M000093520.PDF",
            url: "data/Certificates of Occupancy/161 W. 56th Street/M000093520.PDF",
            icon: FileTextIcon,
          },
        ],
      },
      {
        title: "152 W. 57th Street",
        url: "data/Certificates of Occupancy/152 W. 57th Street",
        items: [
          {
            title: "M00103924B.PDF",
            url: "data/Certificates of Occupancy/152 W 57th Street/M00103924B.PDF",
            icon: FileTextIcon,
          },
        ],
      },
    ],
  },
];

export const DATABASES = [
  {
    name: "Job Filings",
    url: "databases/job_filings",
    icon: HardHat,
  },
  {
    name: "Complaints",
    url: "databases/complaints",
    icon: Pointer,
  },
  {
    name: "Violations",
    url: "databases/violations",
    icon: TriangleAlert,
  },
];

interface SelectedFileState {
  filePath: string | null;
  pageNumber: number | null;
  setSelectedFile: (path: string | null, pageNumber?: number) => void;
}

export const useSelectedFile = create<SelectedFileState>((set) => ({
  filePath: null,
  pageNumber: null,
  setSelectedFile: (path, pageNumber) =>
    set({ filePath: path, pageNumber: pageNumber ?? null }),
}))

