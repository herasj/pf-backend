import { UsersService } from './users.service';
import { Controller, Get, Param } from '@nestjs/common';
import { ApiOperation, ApiTags } from '@nestjs/swagger';

@ApiTags('Users')
@Controller()
export class UsersController {
  constructor(private readonly userService: UsersService) {}

  @ApiOperation({ summary: 'Get common users' })
  @Get('common')
  async getCommon() {
    return await this.userService.getCommonUsers();
  }

  @ApiOperation({ summary: 'Get user details' })
  @Get(':id/details')
  async getDetails(@Param('id') id: string) {
    return await this.userService.getUserDetails(id);
  }

  @ApiOperation({ summary: 'Get user details by username' })
  @Get('username/:username/details')
  async getDetailsByUsername(@Param('username') username: string) {
    return await this.userService.getUserDetailsByUsername(username);
  }
}
