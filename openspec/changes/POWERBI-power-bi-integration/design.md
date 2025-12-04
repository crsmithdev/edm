# Design: Power BI Integration

## Context from Discussion

User requested integration with Microsoft Power BI for visualization of EDM analysis results. This captures the initial exploration of the requirement without deep technical planning.

## Key Decisions

1. **Export Format**: Use Power BI's native data sources (CSV, Excel, or direct API if available)
2. **Data Structure**: Transform nested JSON analysis results into tabular format suitable for Power BI
3. **Template Strategy**: Provide pre-built `.pbix` templates for common visualizations
4. **CLI Integration**: Add export flag to `edm analyze` command for seamless workflow

## Open Questions

- What specific visualizations are most valuable? (energy curves, structural timeline, beat grid, etc.)
- Should this support batch export for multiple tracks?
- Local-only export or also Power BI Service (cloud) integration?
- What's the preferred intermediate format - CSV, Excel, parquet?
- Do we need real-time streaming updates or static export is sufficient?

## Rationale

Power BI chosen for its:
- Wide adoption in data analysis workflows
- Strong time-series visualization capabilities
- Interactive dashboard features
- Template sharing for consistent reports

Alternative considered: Custom web dashboard (more control but higher development cost)

## Next Steps

Expand this proposal with `/os propose POWERBI` after clarifying requirements and approach.
