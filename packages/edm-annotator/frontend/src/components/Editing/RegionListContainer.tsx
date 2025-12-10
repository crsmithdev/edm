import { useStructureStore } from "@/stores";
import { Card } from "@/components/UI";
import { RegionList } from "./RegionList";
import { SaveButton } from "./SaveButton";

/**
 * Container for regions list with header
 */
export function RegionListContainer() {
  const { regions } = useStructureStore();

  return (
    <Card
      padding="sm"
      style={{
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        maxHeight: "600px",
        padding: 0,
      }}
    >
      <div
        style={{
          padding: "var(--space-4)",
          borderBottom: "1px solid var(--border-primary)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <h3
          style={{
            color: "var(--text-primary)",
            fontSize: "var(--font-size-lg)",
            fontWeight: "var(--font-weight-semibold)",
            margin: 0,
          }}
        >
          Regions
        </h3>
        <div style={{ width: "auto" }}>
          <SaveButton />
        </div>
      </div>

      {regions.length === 0 ? (
        <div
          style={{
            padding: "var(--space-6)",
            textAlign: "center",
            color: "var(--text-muted)",
            fontSize: "var(--font-size-sm)",
          }}
        >
          No regions defined
        </div>
      ) : (
        <RegionList />
      )}
    </Card>
  );
}
