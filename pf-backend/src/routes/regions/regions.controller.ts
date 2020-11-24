import { Controller, Get, Param, Query } from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';
import { RegionsService } from './regions.service';

@Controller('region')
@ApiTags('Regions')
export class RegionsController {
  constructor(private readonly regionService: RegionsService) {}

  @ApiOperation({ summary: 'Get common regions' })
  @Get('common')
  async getCommon() {
    return await this.regionService.getCommonRegions();
  }

  @ApiOperation({ summary: 'Get region details by name' })
  @Get(':name/details')
  async getDetailsByName(@Param('name') name: string) {
    return await this.regionService.getRegionDetailsByName(name);
  }

  @ApiOperation({ summary: 'Autocomplete search bar' })
  @Get('search')
  async autocomplete(@Query('name') name: string) {
    return await this.regionService.autoCompleteRegion(name);
  }

  @ApiOperation({ summary: 'Get region details by latitude and longitude' })
  @Get('locate')
  async getRegionDetailsByLatLon(
    @Query('lat') lat: number,
    @Query('long') long: number,
  ) {
    return await this.regionService.getRegionDetailsByLatLon(
      Number(lat),
      Number(long),
    );
  }
}
